"""
circuit_engine_v2.py
─────────────────────────────────────────────────────────────────────────────
회로 해석 백엔드 엔진  ―  A · B · C 파트 통합 모듈  (구조적 변화 확장형 v2)

  ┌─ A파트 ─────────────────────────────────────────────────────────────────┐
  │  solve_node_voltages(initial_circuit)                                   │
  │    SymPy 마디 해석법으로 노드 전압 수치 해를 반환                        │
  │  apply_solution_to_circuit(circuit, solution)                           │
  │    해를 회로 JSON 구조에 반영 (node_voltage, terminal_voltages 갱신)    │
  ├─ B파트 ─────────────────────────────────────────────────────────────────┤
  │  calculate_currents_and_directions(circuit_with_voltages)               │
  │    전압이 채워진 회로를 순회하며                                          │
  │    current_value(양수) · current_direction(from→to) · is_changed 주입  │
  ├─ C파트  [v2 구조적 변화 확장형] ────────────────────────────────────────┤
  │                                                                         │
  │  StepDef  ― 단일 풀이 단계의 완전한 명세 dataclass                      │
  │    .title        단계 제목                                               │
  │    .description  풀이 설명 (LaTeX 가능)                                  │
  │    .circuit      이 단계 완료 시점의 회로 JSON                            │
  │                  ★ 토폴로지 변형(소자 삭제·합성)이 이미 반영된 객체      │
  │    .theory_ids   사용된 이론 ID 목록                                     │
  │                                                                         │
  │  build_steps(step_defs)                                                 │
  │    StepDef 리스트를 받아 step_id·step_index 자동 부여 후                 │
  │    각 StepDef.circuit을 circuit_snapshot에 deepcopy 주입                │
  │    → 어떤 토폴로지 변형도 스냅샷에 그대로 보존됨                          │
  │                                                                         │
  │  build_applied_theories(steps, initial_circuit, solution, result)       │
  │    교과서 단원 순으로 applied_theories[] 생성 + linked_step_id 주입      │
  │    description: 해당 회로 스냅샷의 물리적 맥락을 서술하는 텍스트          │
  │                                                                         │
  │  run_full_pipeline(initial_circuit)                                     │
  │    A→B→C 전체 파이프라인 실행. 내부에서 StepDef 리스트를 조립하여        │
  │    build_steps()에 전달.                                                 │
  │                                                                         │
  │  ┌── 향후 확장 진입점 ──────────────────────────────────────────────┐   │
  │  │  AI가 토폴로지 변형 회로를 생성하면:                               │   │
  │  │    step_defs = [                                                  │   │
  │  │      StepDef("원본", ..., circuit=initial),                       │   │
  │  │      StepDef("테브난 변환", ..., circuit=thevenin_circuit),        │   │
  │  │      StepDef("등가 저항", ..., circuit=simplified_circuit),        │   │
  │  │    ]                                                              │   │
  │  │    steps = build_steps(step_defs)  ← 구조 변경 없이 바로 사용     │   │
  │  └───────────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────────────┘

지원 소자 타입
  resistor / independent_current / independent_voltage / CCCS / VCCS

KCL 부호 규칙
  "유출(leaving) − 유입(entering) = 0" 형태로 방정식 구성
  resistor             I=(V_pos−V_neg)/R   pos(+) neg(−)
  independent_current  내부 neg→pos        pos(+) neg(−)
  CCCS / VCCS          I_dep=gain·ctrl     pos(+) neg(−)
"""

from __future__ import annotations

import copy
import textwrap
from dataclasses import dataclass, field
from typing import Any

import sympy
from sympy import Rational, Symbol, solve, symbols


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX 심볼 정규화 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _sym_to_latex(name: str) -> str:
    """
    문자열 심볼명을 KaTeX/MathJax 렌더링 규격 LaTeX 아래첨자 표기로 변환.

    규칙
    ----
    - 알파벳+숫자 패턴(끝부분)을 아래첨자로 변환:
        'n3'  → 'n_{3}'
        'V1'  → 'V_{1}'
        'n12' → 'n_{12}'
    - 숫자 없는 순수 알파벳(GND, Vth, R)은 그대로 보존.
    - 특수값 '0'(GND 심볼)은 그대로 '0' 반환.

    사용 목적
    ---------
    sympy Symbol 객체는 sympy.latex(sym) 으로 처리하고,
    이 함수는 dict key 나 node['label'] 처럼 **문자열로만 존재하는**
    심볼명을 LaTeX 아래첨자로 변환할 때 사용한다.

    Examples
    --------
    >>> _sym_to_latex('n3')   # 'n_{3}'
    >>> _sym_to_latex('V1')   # 'V_{1}'
    >>> _sym_to_latex('GND')  # 'GND'
    >>> _sym_to_latex('0')    # '0'
    """
    import re
    return re.sub(r'([A-Za-z]+)(\d+)$', r'\1_{\2}', name)


# ═════════════════════════════════════════════════════════════════════════════
# C파트 핵심 데이터 명세 — StepDef
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class StepDef:
    """
    단일 풀이 단계의 완전한 명세 객체.

    build_steps()의 유일한 입력 단위이며, 외부(AI·파이프라인)에서
    주입하는 방식으로 확장된다.

    Fields
    ------
    title       : 단계 제목 (UI 헤더)
    description : 풀이 설명 — 수식(LaTeX) 포함 가능
    circuit     : 이 단계 완료 시점의 회로 JSON dict
                  ★ 토폴로지 변형(소자 삭제·합성·교체)이 이미 반영된 객체.
                    build_steps()는 이 객체를 deepcopy하여 circuit_snapshot에 저장.
    theory_ids  : 이 단계에 사용된 이론 ID 목록 (applied_theories 역방향 참조용)

    Usage
    -----
    # 마디 해석 단계 (숫자만 채우는 기존 방식)
    StepDef(title="KCL 방정식 수립", description="...", circuit=initial_circuit,
            theory_ids=["th_kcl"])

    # 테브난 변환 단계 (AI가 토폴로지를 변형한 새 회로를 주입)
    StepDef(title="테브난 등가 변환", description="...", circuit=thevenin_circuit,
            theory_ids=["th_thevenin"])
    """
    title       : str
    description : str
    circuit     : dict
    theory_ids  : list[str] = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════════
# 공통 내부 헬퍼
# ═════════════════════════════════════════════════════════════════════════════

def _build_symbol_map(nodes: list[dict]) -> dict[str, Any]:
    """node_id → SymPy Symbol(미지수) 또는 Integer(0, 접지)

    [BUG-1 방어] LLM이 node["symbol"]을 null/빈 문자열로 생성하면
    symbols() 호출 시 ValueError가 발생한다.
    - None  → node_id 자체를 심볼 이름으로 폴백
    - ""    → 동일하게 node_id 폴백
    - "0"   → 접지(GND), sympy.Integer(0) 고정
    """
    sym_map: dict[str, Any] = {}
    for node in nodes:
        nid      = node["node_id"]
        sym_str  = node.get("symbol")                # None-safe 접근

        # ── null / 빈 문자열 폴백 ──────────────────────────────────────────
        if not sym_str:                              # None, "", 0(falsy 숫자) 모두 처리
            sym_str = nid                            # node_id를 심볼 이름으로 대체
            node["symbol"] = sym_str                 # ★ circuit dict 에도 동기화 (apply_solution 참조)
            print(
                f"[경고][_build_symbol_map] 노드 '{nid}'의 symbol이 "
                f"'{node.get('symbol', None)!r}'입니다. '{sym_str}'으로 폴백합니다."
            )

        sym_map[nid] = (
            sympy.Integer(0) if sym_str == "0" else symbols(sym_str, real=True)
        )
    return sym_map


def _get_sym(sym_map: dict, node_id: str) -> Any:
    return sym_map.get(node_id, sympy.Integer(0))


def _unknowns(sym_map: dict) -> list[Symbol]:
    return [v for v in sym_map.values() if isinstance(v, Symbol)]


def _build_control_expr(
    control_variable: dict,
    element_map: dict[str, dict],
    sym_map: dict,
) -> Any:
    """CCCS / VCCS 제어량 → SymPy 식"""
    src_id = control_variable["source_element_id"]
    src_el = element_map.get(src_id)
    if src_el is None:
        raise KeyError(f"제어 소자 '{src_id}'가 elements 목록에 없습니다.")

    V_pos  = _get_sym(sym_map, src_el["nodes"]["positive_node_id"])
    V_neg  = _get_sym(sym_map, src_el["nodes"]["negative_node_id"])
    v_diff = V_pos - V_neg

    ctrl_type = control_variable["type"]
    if ctrl_type == "current":
        if src_el["type"] != "resistor":
            raise ValueError(
                f"CCCS 제어 소자 '{src_id}'는 저항이어야 합니다 "
                f"(현재 type='{src_el['type']}')."
            )

        R_raw = src_el["value"]

        if R_raw == 0:
            raise ValueError(
                f"CCCS 제어 저항 '{src_id}'의 저항값이 0Ω입니다. "
                "전류 제어식을 계산할 수 없습니다."
            )

        R = Rational(src_el["value"]).limit_denominator(10**9)
        return v_diff / R

    elif ctrl_type == "voltage":
        return v_diff
    else:
        raise ValueError(f"지원하지 않는 control_variable.type: '{ctrl_type}'")


def _build_kcl_equations(
    elements: list[dict],
    nodes: list[dict],
    sym_map: dict,
) -> list[Any]:
    element_map = {el["element_id"]: el for el in elements}
    non_gnd = [nd for nd in nodes if nd["symbol"] != "0"]
    kcl_buf: dict[str, Any] = {nd["node_id"]: sympy.Integer(0) for nd in non_gnd}
    supernode_pairs: list[tuple[str, str, Any]] = []

    # 1. KCL 전류 정보 수집
    for el in elements:
        etype   = el["type"].strip()
        pos_nid = el["nodes"]["positive_node_id"]
        neg_nid = el["nodes"]["negative_node_id"]
        V_pos   = _get_sym(sym_map, pos_nid)
        V_neg   = _get_sym(sym_map, neg_nid)

        if etype == "resistor":
            R = Rational(el["value"]).limit_denominator(10**9)
            I = (V_pos - V_neg) / R
            if pos_nid in kcl_buf: kcl_buf[pos_nid] += I
            if neg_nid in kcl_buf: kcl_buf[neg_nid] -= I

        elif etype == "independent_current":
            Is = Rational(el["value"]).limit_denominator(10**9)
            if pos_nid in kcl_buf: kcl_buf[pos_nid] += Is
            if neg_nid in kcl_buf: kcl_buf[neg_nid] -= Is

        elif etype == "independent_voltage":
            supernode_pairs.append(
                (pos_nid, neg_nid, Rational(el["value"]).limit_denominator(10**9))
            )

        elif etype in ("CCCS", "VCCS"):
            gain  = Rational(el["value"]).limit_denominator(10**9)
            ctrl  = _build_control_expr(el["control_variable"], element_map, sym_map)
            I_dep = gain * ctrl
            if pos_nid in kcl_buf: kcl_buf[pos_nid] += I_dep
            if neg_nid in kcl_buf: kcl_buf[neg_nid] -= I_dep

    # 2. 슈퍼노드 병합: neg 노드의 전류 정보를 pos 노드로 몰아넣기
    #
    # [BUG-1 방어] neg_nid 또는 pos_nid 가 GND(n0)일 수 있다.
    # kcl_buf 는 non_gnd 노드만 키로 가지므로 GND 노드 접근 시 KeyError 발생.
    #
    # 슈퍼노드 원리:
    #   전압원이 pos(Va)–neg(Vb) 사이에 걸리면
    #   - Va–Vb = Vs  (전압 제약, step 4 에서 추가)
    #   - Va 와 Vb 를 하나의 슈퍼노드로 묶어 KCL 합산 후 방정식 1개로 줄임
    #   - neg_nid 가 GND(kcl_buf 에 없음)이면 합산 불필요, pos 방정식만 폐기
    #     (pos 는 전압 제약식이 대신하므로 KCL 방정식 불필요)
    for pos_nid, neg_nid, Vs in supernode_pairs:
        # ── Case A: neg 가 GND ────────────────────────────────────────────
        # 전압 제약식 pos=Vs 만으로 충분. pos KCL 방정식 제거.
        if neg_nid not in kcl_buf:
            if pos_nid in kcl_buf:
                kcl_buf[pos_nid] = None             # pos KCL 폐기(전압 제약이 대신)
            continue

        # ── Case B: pos 가 GND (비정상, 전압원 극성 역전 입력) ───────────────
        if pos_nid not in kcl_buf:
            kcl_buf[neg_nid] = None                 # neg KCL 폐기
            continue

        # ── Case C: pos, neg 둘 다 비접지 ────────────────────────────────
        kcl_buf[pos_nid] += kcl_buf[neg_nid]        # neg 전류 정보를 pos 로 합산
        kcl_buf[neg_nid] = None                     # neg 방정식 폐기(슈퍼노드 흡수)

    # 3. 방정식 생성 (kcl_buf가 None이 아닌 것만)
    equations = [
        sympy.Eq(kcl_buf[nd["node_id"]], 0) 
        for nd in non_gnd if kcl_buf[nd["node_id"]] is not None
    ]
    
    # 4. 전압원 제약 방정식 추가
    for pos_nid, neg_nid, Vs in supernode_pairs:
        equations.append(sympy.Eq(_get_sym(sym_map, pos_nid) - _get_sym(sym_map, neg_nid), Vs))
    
    return equations



# ═════════════════════════════════════════════════════════════════════════════
# A파트  공개 API
# ═════════════════════════════════════════════════════════════════════════════

def solve_node_voltages(initial_circuit: dict) -> dict[str, float]:
    """
    A파트 메인 함수.
    initial_circuit JSON → SymPy 마디 해석법 → 노드 전압 수치 해.

    Returns
    -------
    { "V1": 20.0, "V2": 10.0, "V3": 10.0, ... }
    """
    nodes, elements = initial_circuit["nodes"], initial_circuit["elements"]
    sym_map         = _build_symbol_map(nodes)
    unknowns        = _unknowns(sym_map)

    print(f"[A-1] 미지수 심볼 동적 생성  : {[str(u) for u in unknowns]}")

    equations = _build_kcl_equations(elements, nodes, sym_map)
    print(f"[A-2] KCL 방정식 조립 ({len(equations)}개):")
    for i, eq in enumerate(equations, 1):
        print(f"       eq{i}: {eq}")
    # ── 디버깅 추가 ──
    A, B = sympy.linear_eq_to_matrix(equations, unknowns)
    print(f"DEBUG - Matrix A: {A}")
    print(f"DEBUG - Vector B: {B}")
    # ───────────────
    
    raw = solve(equations, unknowns, dict=True)
    if not raw:
        raise ValueError(
            "SymPy가 해를 찾지 못했습니다.\n"
            f"  방정식: {equations}\n  미지수: {unknowns}"
        )
    if len(raw) > 1:
        raise ValueError(f"해가 유일하지 않습니다 ({len(raw)}개). 회로를 확인하세요.")

    result = {str(sym): float(val) for sym, val in raw[0].items()}
    print(f"[A-3] sympy.solve() 결과    : {result}")
    return result


def apply_solution_to_circuit(circuit: dict, solution: dict[str, float]) -> dict:
    """
    풀이 결과를 회로 JSON에 반영한다.

    JSON 내 데이터 
    - node_voltage     : 노드 전압
    - terminal_voltage : 소자 단자 전압
    """
    updated    = copy.deepcopy(circuit) # 원본 보호
    sym_to_val = {**solution, "0": 0.0} # 그라운드(접지) 0V 주입

    # 1. 노드 전압 반영 및 terminal_voltages 딕셔너리 깨끗하게 초기화
    node_map = {}
    for node in updated["nodes"]: 
        symbol = node["symbol"]
        # 대소문자나 혹시 모를 누락 방지를 위해 안전하게 get 처리
        voltage = sym_to_val.get(symbol) 
        
        node["node_voltage"] = voltage
        node["terminal_voltages"] = {}  # AI가 보낸 오염된 데이터가 있을 수 있으므로 초기화
        node_map[node["node_id"]] = node

    # 2. elements 구조를 순회하며 소자 연결 관계에 따라 노드별 단자 정보 역주입
    for el in updated["elements"]:
        eid = el["element_id"]
        pos_nid = el["nodes"]["positive_node_id"]
        neg_nid = el["nodes"]["negative_node_id"]

        # positive_node_id 단자 매핑 및 전압 할당
        if pos_nid in node_map:
            node_map[pos_nid]["terminal_voltages"][eid] = {
                "terminal": "positive",
                "voltage": node_map[pos_nid]["node_voltage"]
            }
            
        # negative_node_id 단자 매핑 및 전압 할당
        if neg_nid in node_map:
            node_map[neg_nid]["terminal_voltages"][eid] = {
                "terminal": "negative",
                "voltage": node_map[neg_nid]["node_voltage"]
            }

    return updated


# ═════════════════════════════════════════════════════════════════════════════
# B파트  전류 크기(양수화) 및 화살표 방향 연산
# ═════════════════════════════════════════════════════════════════════════════

def _node_voltage_map(circuit: dict) -> dict[str, float]:
    """
    [BUG-1 방어] node_voltage 가 None 인 경우 예외를 발생시킨다.
    미계산 노드를 0V로 간주하지 않아 잘못된 전류 계산을 방지한다.
    """
    vmap: dict[str, float] = {}

    for nd in circuit["nodes"]:
        raw = nd.get("node_voltage")

        if raw is None:
            raise ValueError(
                f"[_node_voltage_map] 노드 '{nd['node_id']}'의 node_voltage 가 None 입니다. "
                f"먼저 apply_solution_to_circuit()를 호출하여 노드 전압을 계산하세요."
            )

        vmap[nd["node_id"]] = float(raw)

    return vmap


def _directed_current(
    signed_current: float,
    pos_nid: str,
    neg_nid: str,
) -> tuple[float, str, str]:
    """
    부호 있는 전류 → (양수 크기, from_node_id, to_node_id)

    signed_current ≥ 0  →  실제 흐름이 pos→neg  : from=pos, to=neg
    signed_current < 0  →  실제 흐름이 neg→pos  : from=neg, to=pos, 크기=abs()
    """
    if signed_current >= 0:
        return abs(signed_current), pos_nid, neg_nid
    else:
        return abs(signed_current), neg_nid, pos_nid

def _voltage_source_current_from_resistor(
    resistor: dict,
    shared_node_id: str,
    is_positive_terminal: bool,
) -> float:
    """
    직렬 저항 전류로부터 전압원 전류(signed current)를 복원한다.

    Parameters
    ----------
    resistor :
        직렬 연결된 저항 element dict

    shared_node_id :
        전압원과 저항이 공유하는 노드 ID

    is_positive_terminal :
        True  → 공유 노드가 전압원 양극
        False → 공유 노드가 전압원 음극

    Returns
    -------
    signed_I :
        전압원 기준 signed current
        (+) pos → neg
        (-) neg → pos
    """

    I = float(resistor["current_value"])

    direction = resistor["current_direction"]

    resistor_leaves_shared_node = (
        direction["from_node_id"] == shared_node_id
    )

    # 공유 노드가 전압원 양극
    if is_positive_terminal:

        if resistor_leaves_shared_node:
            return -I
        else:
            return I

    # 공유 노드가 전압원 음극
    else:

        if resistor_leaves_shared_node:
            return I
        else:
            return -I

def calculate_currents_and_directions(circuit_with_voltages: dict) -> dict:
    """
    B파트 메인 함수.
    전압이 채워진 회로 JSON을 받아 모든 소자에
    current_value(항상 양수) · current_direction(from→to)
    를 계산·주입한 새 회로 객체를 반환. (원본 불변)
    """
    result   = copy.deepcopy(circuit_with_voltages) # A파트의 apply 함수가 대입됨
    vmap     = _node_voltage_map(result)
    el_map   = {el["element_id"]: el for el in result["elements"]}

    voltage_sources: list[dict] = []

    for el in result["elements"]:
        etype   = el["type"]
        pos_nid = el["nodes"]["positive_node_id"]
        neg_nid = el["nodes"]["negative_node_id"]
        Vp      = vmap[pos_nid]
        Vn      = vmap[neg_nid]

        # ── 저항: I = (V_pos − V_neg) / R ────────────────────────────────
        if etype == "resistor":
            R = float(el["value"])

            if R == 0:
                raise ValueError(
                    f"저항 '{el['element_id']}' 의 저항값이 0Ω 입니다."
                )

            signed_I = (Vp - Vn) / R

        # ── 독립 전류원: 내부 neg→pos, pos 단자 유출 방향 기준 ─────────────
        elif etype == "independent_current":
            signed_I = float(el["value"])

        # ── 독립 전압원: 슈퍼노드 전류는 별도 계산 대상 (placeholder) ──────
        elif etype == "independent_voltage":
            voltage_sources.append(el)
            continue

        # ── CCCS: I_dep = β · Ix,  Ix = (V_src_pos − V_src_neg) / R ──────
        elif etype == "CCCS":
            beta   = float(el["value"])
            ctrl   = el["control_variable"]

            src_id = ctrl["source_element_id"]

            if src_id not in el_map:
                raise ValueError(
                    f"CCCS 제어소자 '{src_id}' 를 찾을 수 없습니다."
                )
            
            src_el = el_map[src_id]

            src_Vp = vmap[src_el["nodes"]["positive_node_id"]]
            src_Vn = vmap[src_el["nodes"]["negative_node_id"]]
            
            if src_el["type"] == "resistor" :
                Ix     = (
                    (src_Vp - src_Vn) / float(src_el["value"])
                    if ctrl["type"] == "current"
                    else (src_Vp - src_Vn)
                )

            elif src_el["type"] == "independent_current":
                Ix = float(src_el["value"])

            else :
                raise ValueError (
                    f"지원하지 않는 제어소자 타입"
                    f"'{src_el['type']}'"
                    )

            signed_I = beta * Ix

        # ── VCCS: I_dep = gm · Vx ─────────────────────────────────────────
        elif etype == "VCCS":
            gm     = float(el["value"])
            ctrl   = el["control_variable"]

            src_id = ctrl["source_element_id"]

            if src_id not in el_map:
                raise ValueError(
                    f"VCCS 제어소자 '{src_id}' 를 찾을 수 없습니다."
                )
            
            src_el = el_map[src_id]


            src_Vp = vmap[src_el["nodes"]["positive_node_id"]]
            src_Vn = vmap[src_el["nodes"]["negative_node_id"]]
            signed_I = gm * (src_Vp - src_Vn)

        else:
            print(f"[경고] B파트 미지원 소자 '{etype}' (id={el['element_id']}) — 스킵")
            continue

        # ── 방향 정제: 부호 판별 → 양수화 + from/to 확정 ──────────────────
        current_value, from_nid, to_nid = _directed_current(
            signed_I, pos_nid, neg_nid
        )
        el["current_value"]     = round(current_value, 10)
        el["current_direction"] = {"from_node_id": from_nid, "to_node_id": to_nid}

    # ───────────────────────────────────────────────
    # Pass 2-1 : 독립 전압원 전류 계산 (직렬연결 - 저항소자 이용)
    # ───────────────────────────────────────────────

    for vs in voltage_sources:
 
        vs_id   = vs["element_id"]
        pos_nid = vs["nodes"]["positive_node_id"]
        neg_nid = vs["nodes"]["negative_node_id"]
 
        # ── Step 1: 새로운 리스트 정의 ────────────────────────────
        pos_resistors: list[dict] = []
        neg_resistors: list[dict] = []
 
        # ── Step 2: 양극/음극 노드에 연결된 저항 소자 수집 ─────────
        for el in result["elements"]:
 
            if el["element_id"] == vs_id:
                continue
 
            el_pos = el["nodes"]["positive_node_id"]
            el_neg = el["nodes"]["negative_node_id"]
 
            if el_pos == pos_nid or el_neg == pos_nid:
                pos_resistors.append(el)
 
            if el_pos == neg_nid or el_neg == neg_nid:
                neg_resistors.append(el)
 
        # ── Step 3 & 4 & 5: len()으로 직렬/병렬 판단 후 처리 ──────
 
        pos_resistor = None
        neg_resistor = None
 
        # 양극 측 판단
        if len(pos_resistors) == 1:
            # 직렬 연결: 해당 저항으로 전압원 전류 계산 가능
            pos_resistor = pos_resistors[0]
        # len > 1 (병렬): pos_resistor = None 유지, 음극 측에서 계산
 
        # 음극 측 판단
        if len(neg_resistors) == 1:
            # 직렬 연결: 해당 저항으로 전압원 전류 계산 가능
            neg_resistor = neg_resistors[0]
        # len > 1 (병렬): neg_resistor = None 유지
 
        # ───────────────────────────────────────────
        # Case 1 : 양극·음극 모두 직렬 저항 존재
        # ───────────────────────────────────────────

        if pos_resistor is not None and neg_resistor is not None:

            signed_I_pos = _voltage_source_current_from_resistor(
                pos_resistor,
                pos_nid,
                True,
            )

            signed_I_neg = _voltage_source_current_from_resistor(
                neg_resistor,
                neg_nid,
                False,
            )

            if abs(signed_I_pos - signed_I_neg) > 1e-9:
                raise ValueError(
                    f"독립전압원 '{vs_id}' 양측 저항으로부터 "
                    f"계산된 전류가 일치하지 않습니다. "
                    f"({signed_I_pos} A vs {signed_I_neg} A)"
                )

            signed_I = signed_I_pos

        # ───────────────────────────────────────────
        # Case 2 : 양극 측 직렬 저항만 존재
        # ───────────────────────────────────────────

        elif pos_resistor is not None:

            signed_I = _voltage_source_current_from_resistor(
                pos_resistor,
                pos_nid,
                True,
            )

        # ───────────────────────────────────────────
        # Case 3 : 음극 측 직렬 저항만 존재
        # ───────────────────────────────────────────

        elif neg_resistor is not None:

            signed_I = _voltage_source_current_from_resistor(
                neg_resistor,
                neg_nid,
                False,
            )

        # ───────────────────────────────────────────
        # Case 4 : 직렬 저항 없음
        # ───────────────────────────────────────────

        else:

            raise NotImplementedError(
                f"독립전압원 '{vs_id}' 와 직렬 연결된 "
                f"저항을 찾을 수 없습니다."
            )

        # ── 부호 → 방향 변환 ───────────────────────────────

        current_value, from_nid, to_nid = _directed_current(
            signed_I,
            pos_nid,
            neg_nid,
        )

        vs["current_value"] = round(current_value, 10)

        vs["current_direction"] = {
            "from_node_id": from_nid,
            "to_node_id": to_nid,
        }

    return result


# ═════════════════════════════════════════════════════════════════════════════
# C파트  단계별 풀이 로그(steps) 및 교과서 이론 매핑(applied_theories) 생성
# ═════════════════════════════════════════════════════════════════════════════

def build_steps(step_defs: list[StepDef]) -> list[dict]:
    """
    C파트 — StepDef 리스트 → steps[] 배열 변환.

    [v2 구조적 변화 확장형]

    기존 방식과의 차이
    ------------------
    - 이전: build_steps(initial_circuit, equations, sym_map, ...) 처럼
      내부에서 고정된 회로 상태를 직접 참조해 스냅샷을 결정했음.
      → 소자 구조가 고정된 채 숫자만 달라지는 회로만 표현 가능.

    - 현재: 각 StepDef.circuit 이 이미 "그 단계에서 보여줄 회로"이므로
      build_steps()는 step_id/step_index 부여 + deepcopy만 담당.
      → 테브난·노턴 변환처럼 소자가 삭제·합성되는 토폴로지 변형도
        스냅샷에 있는 그대로 보존됨.

    Parameters
    ----------
    step_defs : list[StepDef]
        외부(AI 또는 run_full_pipeline)에서 주입하는 단계 명세 목록.
        순서가 곧 step_index 순서.

    Returns
    -------
    list[dict]  — steps[] 배열 (JSON 직렬화 가능)
        각 원소:
          step_id            str   "step_1", "step_2", ...
          step_index         int   0-based
          title              str
          description        str
          circuit_snapshot   dict  StepDef.circuit의 deepcopy
          applied_theory_ids list[str]
    """
    steps: list[dict] = []
    for idx, sd in enumerate(step_defs, start=1):
        steps.append({
            "step_id"            : f"step_{idx}",
            "step_index"         : idx - 1,
            "title"              : sd.title,
            "description"        : textwrap.dedent(sd.description).strip(),
            "circuit_snapshot"   : copy.deepcopy(sd.circuit),
            "applied_theory_ids" : list(sd.theory_ids),
        })
    return steps


def _assemble_step_defs(
    initial_circuit : dict,
    equations       : list[Any],
    sym_map         : dict,
    unknowns        : list[Symbol],
    solution        : dict[str, float],
    circuit_with_v  : dict,
    result_circuit  : dict,
) -> list[StepDef]:
    """
    run_full_pipeline() 내부 전용 헬퍼.

    마디 해석법 파이프라인의 각 연산 단계를 StepDef 목록으로 조립.
    AI가 외부에서 StepDef를 주입하면 이 함수 대신 AI 생성 목록을
    build_steps()에 직접 전달하면 된다.

    토폴로지 변형 단계를 추가하려면 이 목록에 원하는 시점에
    StepDef(circuit=변형된_회로, ...) 를 삽입하면 됨.
    """
    step_defs: list[StepDef] = []

    # ── 공통 표현 ──────────────────────────────────────────────────────────
    sym_decl  = ", ".join(sympy.latex(u) for u in unknowns)  # [PATCH 2] str→latex
    gnd_node  = next((nd for nd in initial_circuit["nodes"] if nd["symbol"] == "0"), None)
    gnd_label = gnd_node["label"] if gnd_node else "GND"

    # ── STEP 1: 노드 설정 및 접지 선택 ────────────────────────────────────
    step_defs.append(StepDef(
        title       = "노드 설정 및 접지 선택",
        description = f"""
            회로에서 접지(Ground) 노드를 선정하고 비접지 노드에 전압 기호를 부여.

            \\[ \\text{{접지}}: {gnd_label} = 0\\,\\text{{V}} \\]

            SymPy 미지수 선언:
            \\[ {sym_decl} = \\texttt{{symbols}}('{sym_decl}',\\, \\text{{real}}=\\text{{True}}) \\]

            비접지 노드 목록: {', '.join(sympy.latex(u) for u in unknowns)}  % [PATCH 3]
        """,
        circuit     = initial_circuit,
        theory_ids  = ["th_node_method"],
    ))

    # ── STEP 2: 종속 전원 제어 변수 수식화 ────────────────────────────────
    dep_sources = [
        el for el in initial_circuit["elements"]
        if el["type"] in ("CCCS", "VCCS", "VCVS", "CCVS")
    ]
    if dep_sources:
        dep_lines = []
        for el in dep_sources:
            ctrl   = el.get("control_variable", {})
            src_id = ctrl.get("source_element_id", "?")
            src_el = next(
                (e for e in initial_circuit["elements"] if e["element_id"] == src_id), None
            )
            if src_el and el["type"] == "CCCS":
                p  = src_el["nodes"]["positive_node_id"]
                n  = src_el["nodes"]["negative_node_id"]
                R  = src_el["value"]
                Vp = sym_map.get(p, sympy.Integer(0))
                Vn = sym_map.get(n, sympy.Integer(0))
                # [PATCH 8] Vp/Vn은 sympy Symbol → sympy.latex()로 subscript 변환
                dep_lines.append(
                    f"CCCS \\texttt{{{el['element_id']}}}: "
                    f"$I_x = \\dfrac{{{sympy.latex(Vp)} - {sympy.latex(Vn)}}}{{{R}}}$,\\quad "
                    f"종속 전류 $= {el['value']} \\cdot I_x$"
                )
        dep_desc = "\n".join(dep_lines) if dep_lines else "(종속 전원 없음)"
    else:
        dep_desc = "(본 회로에 종속 전원이 없습니다)"

    step_defs.append(StepDef(
        title       = "종속 전원 제어 변수 수식화",
        description = f"""
            종속 전원의 제어 변수를 노드 전압 기호 식으로 변환하여
            KCL 방정식에 대입 가능한 형태로 정리.

            {dep_desc}
        """,
        circuit     = initial_circuit,
        theory_ids  = ["th_dep_source"],
    ))

    # ── STEP 3: 노드별 KCL 방정식 수립 ────────────────────────────────────
    # [PATCH 4] sympy.latex(eq.lhs) 이미 n_{3}, V_{1} 등 올바른 LaTeX 생성
    # 수식 번호는 \text{eq}_{i} → 일반 eq_{i} 로 단순화
    eq_lines = [
        f"\\mathrm{{eq}}_{{{i}}}: \\quad {sympy.latex(eq.lhs)} = 0"
        for i, eq in enumerate(equations, 1)
    ]
    _sep = " \\\\\
"  # [PATCH 4 fix] LaTeX 줄바꿈 구분자 (\\\n)
    eq_block = _sep.join(eq_lines)

    step_defs.append(StepDef(
        title       = "노드별 KCL 방정식 수립",
        description = f"""
            각 비접지 노드에 KCL(유출전류 합 = 유입전류 합)을 적용하여
            미지수 심볼로 구성된 연립방정식을 조립.

            $$
            \\begin{{cases}}
            {eq_block}
            \\end{{cases}}
            $$
        """,
        circuit     = initial_circuit,
        theory_ids  = ["th_kcl", "th_node_method"],
    ))

    # ── STEP 4: 연립방정식 행렬화 & sympy.solve() 풀이 ────────────────────
    A_mat, b_vec = sympy.linear_eq_to_matrix(
        [eq.lhs - eq.rhs for eq in equations], unknowns
    )
    G_latex  = sympy.latex(A_mat)
    V_latex  = sympy.latex(sympy.Matrix(unknowns))
    b_latex  = sympy.latex(-b_vec)
    # [PATCH 5] solution dict key는 str('V1','n3') → _sym_to_latex()로 LaTeX 변환
    sol_block = ",\\quad ".join(
        f"{_sym_to_latex(sym)} = {val:.4f}\\,\\text{{V}}" for sym, val in solution.items()
    )

    step_defs.append(StepDef(
        title       = "연립방정식 행렬화 및 풀이",
        description = f"""
            조립된 KCL 방정식을 컨덕턴스 행렬 형태 $[G][V]=[I]$ 로 정리 후
            \\texttt{{sympy.solve()}} 로 수치 해를 도출.

            \\[
            {G_latex}
            {V_latex}
            =
            {b_latex}
            \\]

            풀이 결과:
            \\[ {sol_block} \\]
        """,
        circuit     = circuit_with_v,   # ← 전압이 채워진 회로 스냅샷
        theory_ids  = ["th_matrix", "th_node_method"],
    ))

    # ── STEP 5: 소자 전류 계산 및 방향 정제 ──────────────────────────────
    el_lines = []
    for el in result_circuit["elements"]:
        cv  = el.get("current_value", 0.0)
        cd  = el.get("current_direction", {})
        frm = cd.get("from_node_id", "?")
        to  = cd.get("to_node_id",   "?")
        el_lines.append(
            # [PATCH 6] frm/to node_id → _sym_to_latex() 적용
            f"\\texttt{{{el['element_id']}}}\\;({el.get('label','')}):"
            f"\\; I = {cv:.4f}\\,\\text{{A}},\\;"
            f"\\text{{방향: }}{_sym_to_latex(frm)} \\to {_sym_to_latex(to)}"
        )
    el_block = " \\\\\n".join(el_lines)

    step_defs.append(StepDef(
        title       = "소자 전류 계산 및 방향 정제",
        description = f"""
            확정된 노드 전압을 이용해 각 소자 전류를 옴의 법칙으로 계산.
            전류 크기는 절대값(양수)으로 통일하고,
            실제 물리적 흐름 방향을 $\\text{{from}} \\to \\text{{to}}$ 로 확정.

            \\[
            \\begin{{aligned}}
            {el_block}
            \\end{{aligned}}
            \\]
        """,
        circuit     = result_circuit,   # ← 전류·방향이 완전히 확정된 회로 스냅샷
        theory_ids  = ["th_ohm", "th_current_dir"],
    ))

    # ── STEP 6: 최종 결과 정리 ────────────────────────────────────────────
    # [PATCH 7] nd['label'] 이 'n1','n3' 등 raw string → _sym_to_latex() 적용
    node_block = ",\\quad ".join(
        f"{_sym_to_latex(nd['label'])} = {nd.get('node_voltage', 0):.2f}\\,\\text{{V}}"
        for nd in result_circuit["nodes"]
    )

    step_defs.append(StepDef(
        title       = "최종 결과 정리",
        description = f"""
            마디 해석법 완료. 모든 노드 전압과 소자 전류가 확정됨.

            \\[
            \\text{{노드 전압: }}
            {node_block}
            \\]

            각 소자의 \\texttt{{current\\_value}}(양수) 와
            \\texttt{{current\\_direction}}(화살표 방향) 이 result\\_circuit 에 반영됨.
        """,
        circuit     = result_circuit,
        theory_ids  = ["th_node_method", "th_kcl", "th_ohm"],
    ))

    return step_defs


def _describe_node_method(ctx: dict) -> str:
    """
    th_node_method: 마디 해석법 — 회로에서 접지 선정 장면 기반 맥락 설명
    linked → step_1 (노드 설정 스냅샷)
    """
    non_gnd = ctx["non_gnd_nodes"]
    gnd     = ctx["gnd_label"]
    sym_list = ", ".join(nd["symbol"] for nd in non_gnd)
    node_count = len(non_gnd)

    return (
        f"이 회로에서 가장 먼저 해야 할 일은 '기준점'을 정하는 거예요. "
        f"회로도에서 {gnd} 노드를 접지(0 V)로 선택했고, "
        f"나머지 {node_count}개의 비접지 노드({sym_list})가 "
        f"우리가 값을 구해야 할 미지의 전압 변수가 됩니다. "
        f"마디 해석법은 이처럼 노드 전압을 미지수로 놓고 "
        f"KCL 방정식 연립계를 세워 한 번에 푸는 강력한 방법이에요. "
        f"접지를 잘 고르면 방정식 수가 최소화되어 계산이 훨씬 편해집니다!"
    )


def _describe_kcl(ctx: dict) -> str:
    """
    th_kcl: KCL — 실제 KCL 방정식이 조립된 step_3 스냅샷 기반 맥락 설명
    linked → step_3 (방정식 수립 스냅샷)
    """
    # 독립 전류원 소자 추출
    i_sources = [
        el for el in ctx["elements"]
        if el["type"] == "independent_current"
    ]
    # 저항 소자 추출
    resistors = [el for el in ctx["elements"] if el["type"] == "resistor"]
    # 전류원 연결 노드 (neg 단자 = 전류 유입 노드)
    inflow_nodes = [
        el["nodes"]["negative_node_id"] for el in i_sources
    ]
    inflow_label = ", ".join(
        nd["label"]
        for nd in ctx["nodes"]
        if nd["node_id"] in inflow_nodes
    )

    is_labels  = ", ".join(el["label"] for el in i_sources)
    is_values  = ", ".join(f"{el['value']} A" for el in i_sources)
    res_labels = ", ".join(el["label"] for el in resistors)

    return (
        f"KCL의 핵심은 '어떤 노드에 들어오는 전류의 합 = 나가는 전류의 합'이에요. "
        f"이 회로의 스냅샷을 보면, 독립 전류원 {is_labels}({is_values})가 "
        f"노드 {inflow_label}로 전류를 공급(유입)하고 있어요. "
        f"그 전류는 {res_labels} 등의 저항들을 통해 각각 다른 노드로 빠져나갑니다. "
        f"엔진은 각 비접지 노드마다 이 관계를 수식으로 표현하여 "
        f"총 {len(ctx['non_gnd_nodes'])}개의 KCL 방정식을 자동으로 조립했습니다. "
        f"방정식의 개수가 미지수의 개수와 딱 맞아야 유일한 해가 존재한다는 점도 기억하세요!"
    )


def _describe_ohm(ctx: dict) -> str:
    """
    th_ohm: 옴의 법칙 — 전류 계산 결과가 담긴 step_5 스냅샷 기반 맥락 설명
    linked → step_5 (전류 계산 스냅샷)
    """
    resistors = [el for el in ctx["elements"] if el["type"] == "resistor"]
    # 결과 회로에서 전류값이 있는 저항만 필터
    computed = [
        el for el in ctx["result_elements"]
        if el["type"] == "resistor" and el.get("current_value") is not None
    ]
    # 가장 직관적인 예시로 첫 번째 저항 선택
    ex = computed[0] if computed else None
    ex_str = ""
    if ex:
        cd  = ex.get("current_direction", {})
        frm = cd.get("from_node_id", "?")
        to  = cd.get("to_node_id",   "?")
        ex_str = (
            f" 예를 들어 {ex.get('label','?')}({ex['value']} Ω)의 경우, "
            f"계산된 전류는 {ex.get('current_value', 0):.4f} A이고 "
            f"실제 전류는 {frm} → {to} 방향으로 흐릅니다."
        )

    res_labels = ", ".join(el["label"] for el in resistors)
    return (
        f"노드 전압을 다 구했으면, 이제 각 소자에 흐르는 전류를 구할 차례예요. "
        f"저항 소자({res_labels})는 옴의 법칙 $I = \\dfrac{{V_{{\\text{{pos}}}} - V_{{\\text{{neg}}}}}}{{R}}$ 을 "  # [PATCH 10]
        f"그대로 적용하면 됩니다. "
        f"전압 차가 양수이면 전류는 pos→neg 방향으로, 음수이면 반대 방향으로 흘러요."
        f"{ex_str} "
        f"이처럼 전압 차의 부호가 전류 방향을 결정하기 때문에, "
        f"엔진은 계산 후 전류를 무조건 양수로 바꾸고 방향 정보를 별도 필드로 저장합니다."
    )


def _describe_dep_source(ctx: dict) -> str:
    """
    th_dep_source: CCCS — 종속 전원 수식화 장면인 step_2 스냅샷 기반 맥락 설명
    linked → step_2 (CCCS 수식화 스냅샷)
    """
    cccs_list = [el for el in ctx["elements"] if el["type"] == "CCCS"]
    if not cccs_list:
        return (
            "종속 전류원(CCCS)은 다른 소자의 전류에 비례하여 전류를 공급하는 소자예요. "
            "이 회로에는 CCCS가 포함되어 있지 않지만, "
            "만약 있다면 제어 저항의 전류를 노드 전압 심볼 식으로 먼저 변환한 뒤 "
            "KCL 방정식에 대입하는 과정이 추가됩니다."
        )

    lines = []
    for cccs in cccs_list:
        ctrl   = cccs.get("control_variable", {})
        src_id = ctrl.get("source_element_id", "?")
        src_el = next(
            (e for e in ctx["elements"] if e["element_id"] == src_id), None
        )
        beta   = cccs["value"]
        label  = cccs.get("label", cccs["element_id"])
        src_label = src_el["label"] if src_el else src_id
        src_R     = src_el["value"] if src_el else "?"
        src_pos   = src_el["nodes"]["positive_node_id"] if src_el else "?"
        src_neg   = src_el["nodes"]["negative_node_id"] if src_el else "?"
        # 심볼 이름 (symbol 필드 참조)
        sym_pos = next(
            (nd["symbol"] for nd in ctx["nodes"] if nd["node_id"] == src_pos), src_pos
        )
        sym_neg = next(
            (nd["symbol"] for nd in ctx["nodes"] if nd["node_id"] == src_neg), src_neg
        )
        # [PATCH 9] sym_pos/neg 를 LaTeX subscript 형태로 변환
        sym_pos_l     = _sym_to_latex(sym_pos)
        sym_neg_disp  = "0" if sym_neg == "0" else sym_neg
        sym_neg_disp_l = _sym_to_latex(sym_neg_disp)

        lines.append(
            f"이 회로에는 종속 전류원 {label}(β={beta})이 있어요. "
            f"'{label}'은 저항 {src_label}({src_R} Ω)을 통과하는 전류 $I_x$를 "
            f"β배 해서 전류를 공급하는 소자입니다. "
            f"$I_x$는 {src_label} 양단의 노드 전압으로 표현하면 "
            f"$I_x = \\dfrac{{{sym_pos_l} - {sym_neg_disp_l}}}{{{src_R}}}$ 이 되고, "
            f"CCCS의 실제 출력 전류는 $\\beta \\cdot I_x = "
            f"\\dfrac{{{beta}({sym_pos_l} - {sym_neg_disp_l})}}{{{src_R}}}$ 로 쓸 수 있어요. "
            f"이 식을 그대로 KCL 방정식에 대입하면 종속 전원도 일반 저항처럼 "
            f"다룰 수 있게 됩니다. 회로 스냅샷에서 {label}이 연결된 노드들을 "
            f"확인하면서 전류 방향을 함께 살펴보세요!"
        )
    return " ".join(lines)


def _describe_matrix(ctx: dict) -> str:
    """
    th_matrix: 행렬 표현 — 전압 해가 채워진 step_4 스냅샷 기반 맥락 설명
    linked → step_4 (행렬 풀이 스냅샷)
    """
    n        = len(ctx["non_gnd_nodes"])
    sym_list = ", ".join(nd["symbol"] for nd in ctx["non_gnd_nodes"])
    sol_str  = ", ".join(
        f"{k} = {v:.2f} V" for k, v in ctx["solution"].items()
    )

    return (
        f"KCL 방정식을 다 세웠으면 이제 연립방정식을 풀어야 해요. "
        f"이 회로의 비접지 노드가 {n}개이므로 미지수도 {n}개({sym_list})이고, "
        f"방정식도 {n}개입니다. "
        f"엔진은 이 연립방정식을 컨덕턴스 행렬 [G], "
        f"노드 전압 벡터 [V], 전류원 벡터 [I]로 구조화한 뒤 "
        f"[G][V] = [I] 형태로 만들고 SymPy의 solve()로 한 번에 풀었어요. "
        f"회로 스냅샷에는 풀이 결과인 {sol_str}이 "
        f"각 노드의 node_voltage 필드에 채워져 있는 걸 확인할 수 있어요. "
        f"행렬 표현을 쓰면 노드 수가 늘어나도 코드 구조를 바꾸지 않고 "
        f"동일한 알고리즘으로 자동 확장되는 장점이 있습니다!"
    )


def _describe_current_dir(ctx: dict) -> str:
    """
    th_current_dir: 전류 방향 관례 — 방향 정제 결과가 담긴 step_5 스냅샷 기반 맥락 설명
    linked → step_5 (전류 방향 확정 스냅샷)
    """
    # 방향이 뒤집힌 소자 (pos→neg 기본 방향과 다른 경우) 찾기
    flipped = []
    for el in ctx["result_elements"]:
        cd  = el.get("current_direction", {})
        frm = cd.get("from_node_id")
        pos = el["nodes"]["positive_node_id"]
        if frm and frm != pos and el.get("current_value", 0) > 0:
            flipped.append(el.get("label", el["element_id"]))

    # 전류가 0인 소자 (브릿지 평형)
    zero_els = [
        el.get("label", el["element_id"])
        for el in ctx["result_elements"]
        if el.get("current_value", 0) == 0
    ]

    flipped_str = (
        f"이 회로에서는 {', '.join(flipped)} 소자의 전류 방향이 "
        f"초기 설정(pos→neg)과 반대로 뒤집혀 확정되었어요. "
        if flipped
        else "이 회로에서는 모든 소자의 전류 방향이 초기 설정(pos→neg)과 동일하게 확정되었어요. "
    )
    zero_str = (
        f"특히 {', '.join(zero_els)}에는 전류가 0 A로, "
        f"브릿지 평형 상태임을 알 수 있어요. "
        if zero_els
        else ""
    )

    return (
        f"전류 계산 결과를 UI에 표시할 때 가장 중요한 규칙이 있어요: "
        f"전류 크기는 반드시 양수여야 하고, 방향은 별도 필드로 관리해야 한다는 거예요. "
        f"수식으로 구한 전류가 음수라는 건 '초기에 설정한 방향과 실제 방향이 반대'라는 뜻일 뿐이에요. "
        f"{flipped_str}"
        f"{zero_str}"
        f"엔진은 이 로직을 _directed_current() 함수 한 곳에서 처리하여, "
        f"어떤 소자 타입이 추가되더라도 동일한 규칙이 일관되게 적용됩니다. "
        f"current_direction(from→to) 정보만 저장됩니다."
    )


def build_applied_theories(
    steps          : list[dict],
    initial_circuit: dict,
    solution       : dict[str, float],
    result_circuit : dict,
) -> list[dict]:
    """
    C파트 — 교과서 단원 순 applied_theories[] 생성.

    각 이론 객체의 description 은 교과서적 정의가 아니라
    해당 단계의 회로 스냅샷에서 '이 공식이 실제로 어떻게 적용되는지'를
    과외 선생님 어투로 서술한 맥락 인식 텍스트입니다.

    Parameters
    ----------
    steps           : build_steps() 가 반환한 steps 배열
    initial_circuit : 원본 회로 (소자·노드 구조 참조용)
    solution        : 노드 전압 수치 해 { "V1": 20.0, ... }
    result_circuit  : 전류·방향까지 확정된 최종 회로
    """
    # ── 공통 컨텍스트 ─────────────────────────────────────────────────────────
    nodes     = initial_circuit["nodes"]
    elements  = initial_circuit["elements"]
    non_gnd   = [nd for nd in nodes if nd["symbol"] != "0"]
    gnd_node  = next((nd for nd in nodes if nd["symbol"] == "0"), None)
    gnd_label = gnd_node["label"] if gnd_node else "GND"

    ctx = {
        "nodes"           : nodes,
        "elements"        : elements,
        "non_gnd_nodes"   : non_gnd,
        "gnd_label"       : gnd_label,
        "solution"        : solution,
        "result_elements" : result_circuit["elements"],
    }

    # ── theory_id별 맥락 설명 빌더 함수 매핑 ──────────────────────────────────
    DESC_BUILDERS = {
        "th_node_method" : _describe_node_method,
        "th_kcl"         : _describe_kcl,
        "th_ohm"         : _describe_ohm,
        "th_dep_source"  : _describe_dep_source,
        "th_matrix"      : _describe_matrix,
        "th_current_dir" : _describe_current_dir,
    }

    # ── steps에서 theory_id 첫 등장 step_id 추적 ─────────────────────────────
    first_step: dict[str, str] = {}
    for step in steps:
        for tid in step.get("applied_theory_ids", []):
            if tid not in first_step:
                first_step[tid] = step["step_id"]

    # ── 교과서 단원 카탈로그 (formula_latex · chapter_order 고정값) ───────────
    CATALOG_META: list[dict] = [
        {
            "theory_id"    : "th_node_method",
            "chapter_name" : "4.2 마디 해석법 (Nodal Analysis)",
            "formula_latex": r"[G][V] = [I]",
            "chapter_order": 1,
        },
        {
            "theory_id"    : "th_kcl",
            "chapter_name" : "2.3 키르히호프 전류 법칙 (KCL)",
            "formula_latex": r"\sum I_{\text{out}} - \sum I_{\text{in}} = 0",
            "chapter_order": 2,
        },
        {
            "theory_id"    : "th_ohm",
            "chapter_name" : "2.1 옴의 법칙",
            "formula_latex": r"I = \frac{V_{\text{pos}} - V_{\text{neg}}}{R}",
            "chapter_order": 3,
        },
        {
            "theory_id"    : "th_dep_source",
            "chapter_name" : "4.1 종속 전원 — CCCS",
            "formula_latex": r"I_s = \beta \, I_x,\quad I_x = \frac{V_{\text{pos}} - V_{\text{neg}}}{R}",
            "chapter_order": 4,
        },
        {
            "theory_id"    : "th_matrix",
            "chapter_name" : "4.4 노드 전압법 — 행렬 표현",
            "formula_latex": r"[G][V] = [I] \;\Rightarrow\; [V] = [G]^{-1}[I]",
            "chapter_order": 5,
        },
        {
            "theory_id"    : "th_current_dir",
            "chapter_name" : "2.2 전류 방향 관례 및 양수화",
            "formula_latex": r"|I| \geq 0,\quad \text{from\_node} \to \text{to\_node}",
            "chapter_order": 6,
        },
    ]

    # ── 최종 조립: 메타 + 맥락 description + linked_step_id ──────────────────
    result: list[dict] = []
    for meta in CATALOG_META:
        tid         = meta["theory_id"]
        builder     = DESC_BUILDERS.get(tid)
        description = builder(ctx) if builder else "(설명 없음)"

        result.append({
            **meta,
            "description"  : description,
            "linked_step_id": first_step.get(tid, steps[-1]["step_id"]),
        })

    result.sort(key=lambda x: x["chapter_order"])
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 전체 파이프라인 통합 실행기
# ═════════════════════════════════════════════════════════════════════════════

def _validate_circuit(circuit: dict) -> None:
    """
    run_full_pipeline() 진입 직전에 호출되는 회로 JSON 사전 검증기.

    [BUG-1 방어] LLM이 생성한 JSON에서 자주 발생하는 스키마 위반을 조기에 감지하여
    SymPy 연산 단계까지 오류가 전파되는 것을 방지한다.

    검사 항목
    ---------
    1. 필수 최상위 키 존재 여부 (nodes, elements)
    2. 각 node 의 필수 필드 및 타입
       - node_id   : str
       - symbol    : str (null 허용 — _build_symbol_map 에서 폴백 처리)
       - terminal_voltages : dict 또는 list (배열이면 경고만, dict 변환은 apply_solution 에서)
    3. 각 element 의 필수 필드 및 타입
       - element_id, type, value, nodes(positive/negative_node_id)
       - value 가 None → 0 으로 교정(경고)
    4. element 의 nodes 참조가 실제 node_id 집합 내에 있는지
    """
    errors:   list[str] = []
    warnings: list[str] = []

    # ── 최상위 키 ───────────────────────────────────────────────────────────
    for key in ("nodes", "elements"):
        if key not in circuit:
            errors.append(f"필수 키 '{key}' 누락")

    if errors:
        raise ValueError(
            "[_validate_circuit] 회로 JSON 구조 오류:\n  " + "\n  ".join(errors)
        )

    nodes    : list[dict] = circuit["nodes"]
    elements : list[dict] = circuit["elements"]
    node_ids : set[str]   = set()

    # ── nodes 검사 ───────────────────────────────────────────────────────────
    for i, nd in enumerate(nodes):
        nid = nd.get("node_id")
        if not nid:
            errors.append(f"nodes[{i}]: 'node_id' 누락 또는 빈 값")
            continue
        node_ids.add(nid)

        if nd.get("symbol") is None:
            warnings.append(f"노드 '{nid}': symbol=null — _build_symbol_map 에서 폴백")

        tv = nd.get("terminal_voltages")
        if tv is not None and not isinstance(tv, (dict, list)):
            errors.append(
                f"노드 '{nid}': terminal_voltages 가 dict/list 가 아님 "
                f"(type={type(tv).__name__})"
            )

    # ── elements 검사 ────────────────────────────────────────────────────────
    for i, el in enumerate(elements):
        eid = el.get("element_id", f"elements[{i}]")

        for field in ("element_id", "type"):
            if not el.get(field):
                errors.append(f"소자 '{eid}': 필수 필드 '{field}' 누락")

        # value 가 None → 0으로 교정
        if el.get("value") is None:
            warnings.append(f"소자 '{eid}': value=null → 0 으로 교정")
            el["value"] = 0

        nodes_field = el.get("nodes", {})
        for terminal in ("positive_node_id", "negative_node_id"):
            ref = nodes_field.get(terminal)
            if not ref:
                errors.append(f"소자 '{eid}': nodes.{terminal} 누락")
            elif node_ids and ref not in node_ids:
                errors.append(
                    f"소자 '{eid}': nodes.{terminal}='{ref}' 가 "
                    f"nodes 목록에 없음 (존재하는 ID: {sorted(node_ids)})"
                )

    # ── 결과 출력 ────────────────────────────────────────────────────────────
    for w in warnings:
        print(f"[경고][_validate_circuit] {w}")
    if errors:
        raise ValueError(
            "[_validate_circuit] 회로 JSON 유효성 검사 실패:\n  "
            + "\n  ".join(errors)
        )


def run_full_pipeline(
    initial_circuit : dict,
    step_defs_override: list[StepDef] | None = None,
) -> dict:
    try:
        # ── [BUG-1 방어] 진입 직전 회로 JSON 사전 검증 ───────────────────────
        _validate_circuit(initial_circuit)
        nodes, elements = initial_circuit["nodes"], initial_circuit["elements"]

        # ── A파트 ─────────────────────────────────────────────────────────────────
        sym_map   = _build_symbol_map(nodes)
        unknowns  = _unknowns(sym_map)
        equations = _build_kcl_equations(elements, nodes, sym_map)

        raw = solve(equations, unknowns, dict=True)
        if not raw:
            raise ValueError("SymPy 해 없음")
        if len(raw) > 1:
            raise ValueError(f"해 유일하지 않음 ({len(raw)}개)")
        solution = {str(sym): float(val) for sym, val in raw[0].items()}

        circuit_with_v = apply_solution_to_circuit(initial_circuit, solution)

        # ── B파트 ─────────────────────────────────────────────────────────────────
        result_circuit = calculate_currents_and_directions(circuit_with_v)

        # ── C파트 ─────────────────────────────────────────────────────────────────
        if step_defs_override is not None:
            step_defs = step_defs_override
        else:
            step_defs = _assemble_step_defs(
                initial_circuit = initial_circuit,
                equations       = equations,
                sym_map         = sym_map,
                unknowns        = unknowns,
                solution        = solution,
                circuit_with_v  = circuit_with_v,
                result_circuit  = result_circuit,
            )

        steps = build_steps(step_defs)

        applied_theories = build_applied_theories(
            steps           = steps,
            initial_circuit = initial_circuit,
            solution        = solution,
            result_circuit  = result_circuit,
        )

        return {
            "initial_circuit"         : initial_circuit,
            "result_circuit"          : result_circuit,
            "solution"                : solution,
            "steps"                   : steps,
            "applied_theories"        : applied_theories,
            # ── [BUG-1 패치] 추가 필드 ──────────────────────────────────────
            # AI가 채워 넣을 때까지 None으로 초기화.
            # _run_sub_solution()에서 ai_output 데이터를 주입한다.
            "target_answer"           : None  # 최종 정답 타겟 (예: {"label":"Vth","value":None,"unit":"V"})
        }
    except Exception as e:
        print(f"!!! ENGINE FAILURE: {e}")
        return {
            "initial_circuit"         : initial_circuit,
            "result_circuit"          : initial_circuit,
            "solution"                : {},
            "steps"                   : [],
            "applied_theories"        : [],
            "target_answer"           : None
        }

# ═════════════════════════════════════════════════════════════════════════════
# 진입점 — 전체 파이프라인 실행 및 출력 검증
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Mock 데이터: 확정 브릿지 회로 initial_circuit ─────────────────────────
    mock_initial_circuit = {
        "circuit_id": "initial",
        "elements": [
            {
                "element_id": "e_Is",
                "type": "independent_current", "label": "Is",
                "value": 2, "unit": "A",
                "nodes": {"positive_node_id": "n0", "negative_node_id": "n1"},
                "current_value": None, "current_direction": {},
                "is_changed": False, "origin_sources": [],
                "transform_description": "독립 전류원 2A",
            },
            {
                "element_id": "e_R1",
                "type": "resistor", "label": "R1",
                "value": 10, "unit": "Ω",
                "nodes": {"positive_node_id": "n1", "negative_node_id": "n2"},
                "current_value": None, "current_direction": {},
                "is_changed": False, "origin_sources": [],
                "transform_description": "저항 10Ω",
            },
            {
                "element_id": "e_R2",
                "type": "resistor", "label": "R2",
                "value": 10, "unit": "Ω",
                "nodes": {"positive_node_id": "n1", "negative_node_id": "n3"},
                "current_value": None, "current_direction": {},
                "is_changed": False, "origin_sources": [],
                "transform_description": "저항 10Ω",
            },
            {
                "element_id": "e_R3",
                "type": "resistor", "label": "R3",
                "value": 10, "unit": "Ω",
                "nodes": {"positive_node_id": "n2", "negative_node_id": "n0"},
                "current_value": None, "current_direction": {},
                "is_changed": False, "origin_sources": [],
                "transform_description": "저항 10Ω",
            },
            {
                "element_id": "e_R4",
                "type": "resistor", "label": "R4",
                "value": 10, "unit": "Ω",
                "nodes": {"positive_node_id": "n3", "negative_node_id": "n0"},
                "current_value": None, "current_direction": {},
                "is_changed": False, "origin_sources": [],
                "transform_description": "저항 10Ω",
            },
            {
                "element_id": "e_R5",
                "type": "resistor", "label": "R5",
                "value": 5, "unit": "Ω",
                "nodes": {"positive_node_id": "n2", "negative_node_id": "n3"},
                "current_value": None, "current_direction": {},
                "is_changed": False, "origin_sources": [],
                "transform_description": "브릿지 저항 5Ω",
            },
            {
                "element_id": "e_CCCS",
                "type": "CCCS", "label": "βI_x",
                "value": 3, "unit": "A/A",
                "nodes": {"positive_node_id": "n2", "negative_node_id": "n0"},
                "current_value": None, "current_direction": {},
                "is_changed": False, "origin_sources": [],
                "transform_description": "CCCS β=3, 제어전류=I(e_R5)",
                "control_variable": {"type": "current", "source_element_id": "e_R5"},
            },
        ],
        "nodes": [
            {
                "node_id": "n0", "label": "GND", "symbol": "0",
                "node_voltage": 0, "is_changed": False,
                "terminal_voltages": {},
                "connected_elements": ["e_Is", "e_R3", "e_R4", "e_CCCS"],
            },
            {
                "node_id": "n1", "label": "V1", "symbol": "V1",
                "node_voltage": None, "is_changed": False,
                "terminal_voltages": {
                    "e_Is": {"terminal": "negative", "voltage": None},
                    "e_R1": {"terminal": "positive",  "voltage": None},
                    "e_R2": {"terminal": "positive",  "voltage": None},
                },
                "connected_elements": ["e_Is", "e_R1", "e_R2"],
            },
            {
                "node_id": "n2", "label": "V2", "symbol": "V2",
                "node_voltage": None, "is_changed": False,
                "terminal_voltages": {
                    "e_R1":   {"terminal": "negative", "voltage": None},
                    "e_R3":   {"terminal": "positive", "voltage": None},
                    "e_R5":   {"terminal": "positive", "voltage": None},
                    "e_CCCS": {"terminal": "positive", "voltage": None},
                },
                "connected_elements": ["e_R1", "e_R3", "e_R5", "e_CCCS"],
            },
            {
                "node_id": "n3", "label": "V3", "symbol": "V3",
                "node_voltage": None, "is_changed": False,
                "terminal_voltages": {
                    "e_R2": {"terminal": "negative", "voltage": None},
                    "e_R4": {"terminal": "positive", "voltage": None},
                    "e_R5": {"terminal": "negative", "voltage": None},
                },
                "connected_elements": ["e_R2", "e_R4", "e_R5"],
            },
        ],
    }

    W = 70
    S = "─" * W
    D = "═" * W

    def hdr(text: str) -> None:
        print(f"\n{text:^{W}}")
        print(S)

    print(D)
    print(f"{'circuit_engine_v2  ―  구조적 변화 확장형 파이프라인 검증':^{W}}")
    print(D)

    