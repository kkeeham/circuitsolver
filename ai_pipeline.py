"""
ai_pipeline.py
─────────────────────────────────────────────────────────────────────────────
[5단계] Claude 3.5 Sonnet Vision API 기반 이미지 독해 및 자동 템플릿 분기 파이프라인

처리 흐름
─────────
  이미지(bytes)
    │
    ▼
  [1] Claude Vision — 이미지 독해 & 구조화 JSON 추출
    │   ├─ parsed_question_text  (문항 번호 제거, ** 마크업 없음)
    │   ├─ problem_type          CONVERSION | CALCULATION
    │   └─ sub_solutions[]       메뉴 분기별 initial_circuit + 풀이 정보
    │
    ▼
  [2] 유형별 파이프라인 분기
    │   └─ CALCULATION  → run_full_pipeline(initial_circuit)
    │
    ▼
  [3] 최종 응답 JSON 조립
        {
          problem_type, parsed_question_text,
          sub_solutions: [
            { menu_id, menu_title, initial_circuit,
              result_circuit, solution, steps, applied_theories }
          ]
        }

의존 모듈
─────────
  circuit_engine_v2.py — run_full_pipeline(), , build_steps(), build_applied_theories()
  anthropic            — pip install anthropic
  환경변수 ANTHROPIC_API_KEY 필수
"""

from __future__ import annotations

import base64
import json
import os
import re
import textwrap
from typing import Any

import anthropic

from circuit_engine_v2 import (
    build_applied_theories,
    build_steps,
    run_full_pipeline,
)


# ═════════════════════════════════════════════════════════════════════════════
# Anthropic 클라이언트 싱글턴
# ═════════════════════════════════════════════════════════════════════════════

def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "환경변수 ANTHROPIC_API_KEY 가 설정되지 않았습니다. "
            ".env 파일 또는 시스템 환경변수에 API 키를 등록해 주세요."
        )
    return anthropic.Anthropic(api_key=api_key)


# ═════════════════════════════════════════════════════════════════════════════
# Claude Vision 호출용 마스터 프롬프트
# ═════════════════════════════════════════════════════════════════════════════

_VISION_SYSTEM_PROMPT = textwrap.dedent("""
당신은 대학교 회로이론 교재의 회로도 이미지를 완벽하게 분석하고,
회로 이미지를 분석하여 구조화된 JSON을 생성하는 전문 AI 엔진입니다.
주어진 이미지를 분석하여 아래 JSON 스키마를 **정확히** 따르는 JSON 객체 하나만 출력하세요.
JSON 앞뒤에 마크다운 코드펜스(```json)나 설명 텍스트를 절대 포함하지 마세요.
반드시 유효한 JSON만 출력하세요.

────────────────────────────────────────────────────────────────────────────
출력 JSON 스키마
────────────────────────────────────────────────────────────────────────────
{
  "topology_reasoning": "회로의 소자들과 노드(도선) 연결 상태를 추론하는 과정 서술",
  "parsed_question_text": string,

  "parsed_question_text": string,
  "problem_type": "CONVERSION" | "CALCULATION",
  "sub_solutions": [
    {
      "menu_id": integer,
      "menu_title": string,
      "target_answer": {
        "label": string,
        "value": number | null,
        "unit":  string,
        "description": string
      },
      "initial_circuit": {
        "circuit_id": string,
        "elements": [ <element_object>, ... ],
        "nodes":    [ <node_object>,    ... ]
      },
      "step_defs": [
        {
          "title":      string,
          "description": string,
          "theory_ids": [string, ...],
          "topology_change": {
            "remove_element_ids": [string, ...],
            "description": string
          } | null
        },
        ...
      ]
    },
    ...
  ]
}


────────────────────────────────────────────────────────────────────────────
각 필드 상세 지침
────────────────────────────────────────────────────────────────────────────

[parsed_question_text]
- 이미지 상단(또는 문제 박스)의 지문 텍스트를 OCR로 정확히 읽어 추출하세요.
- '문 14.', '14.', '[문제 3]' 등 문항 번호는 완전히 제거하세요.
- 추출된 순수 문제 텍스트만 출력하세요. ** 같은 마크다운 강조 기호를 절대 사용하지 마세요.
  올바른 예: "그림의 회로에서 테브난 등가회로를 구하시오. 단, Vs=10V, R1=5Ω, R2=10Ω."
  금지된 예: "**그림의 회로에서 테브난 등가회로를 구하시오.**" ← ** 절대 금지

[problem_type]
- "CONVERSION": 회로의 구조(토폴로지)가 단계별로 변하는 문제
  (테브난·노턴 등가 변환, 등가저항 병합, 전원 변환 등)
- "CALCULATION": 회로 형태는 그대로이고 특정 전류·전압 값만 구하는 문제
  (마디 해석법, 망로 해석법, 중첩 정리 등)

[sub_solutions 배열 구성 규칙]
- 지문에 "(1) 테브난 등가회로", "(2) 노턴 등가회로" 처럼 N가지 요구가 있으면 N개 생성
- 요구사항이 하나이거나 CALCULATION이면 배열 길이 = 1

[menu_title]
- 지문의 해당 요구사항 텍스트 그대로: "(1) 테브난 등가회로 변환"

[initial_circuit — element_object 스키마]
{
  "element_id": string,
  "type": "resistor" | "independent_voltage" | "independent_current"
          | "CCCS" | "VCCS" | "VCVS" | "CCVS",
  "label": string,
  "value": number,
  "unit": string,
  "nodes": {
    "positive_node_id": string,
    "negative_node_id": string
  },
  "current_value": null,
  "current_direction": { "from_node_id": null, "to_node_id": null },
  "is_changed": false,
  "origin_sources": [],
  "transform_description": string,
  "control_variable": {
    "type": "current" | "voltage",
    "source_element_id": string
  }
}

[initial_circuit — node_object 스키마]


{
  "node_id": string,
  "label": string,
  "symbol": string,
  "node_voltage": null,
  "terminal_voltages": {},
  "is_changed": false,
  "connected_elements": [string, ...]
}
- ★ 만약 구해야 하는 타겟 변수(예: v_o, I_x 등)가 특정 노드의 전압을 의미한다면, 반드시 해당 노드의 "label" 필드에 그 변수명을 그대로 작성하세요.
- ★ 만약 구해야 하는 타겟 변수(예: I_o, I_x 등)가 특정 소자에 흐르는 전류를 의미한다면, 반드시 해당 소자의 "label" 필드에 그 변수명을 포함시키세요. (예: "8Ω 저항 (I_o)")


────────────────────────────────────────────────────────────────────────────
[매우 중요] "그림"이 아니라 "회로 구조"
────────────────────────────────────────────────────────────────────────────

회로는 그래프(Graph)이다.

node = 전위 영역
element = node와 node를 연결하는 edge

JSON 생성 전
회로를 node-edge 그래프로 변환한 뒤
elements와 nodes를 생성하라.

────────────────────────────────────────────────────────────────────────────
[매우 중요] 소자(element) - 노드(node) 매칭 규칙 
────────────────────────────────────────────────────────────────────────────

노드(node)는 "소자의 단자"가 아니다.

노드는 전기적으로 연속된 도체 영역 전체를 의미한다.


STEP 0.
회로의 모든 소자를 식별하라.

STEP 1.
각 소자의 양단(좌/우 또는 상/하 단자)을 식별하라.

STEP 2.
각 단자가 연결된 전기적 도체 영역을 추적하라.

예)

VS1+
→ n1

VS1-
→ n0

R1 좌측
→ n1

R1 우측
→ n2

STEP 3.
위 결과를 바탕으로 노드(node)를 생성하라.

STEP 4.
각 node의 connected_elements를 생성하라.

STEP 5.
최종 자기검증 (반드시 수행)

JSON 생성 후 모든 element에 대해 다음을 확인하라.

각 element의
positive_node_id,
negative_node_id

가 가리키는 node의 connected_elements 배열에
해당 element_id가 반드시 존재해야 한다.

하나라도 불일치하면 JSON을 다시 생성하라.

STEP 6. 소자 단자 검증

소자 양단이 동일한 두 노드로 반복 생성되었다면
회로를 다시 분석하라.

예)

VS1 : n1 ↔ n0
R1  : n1 ↔ n0

가 발생했을 때

실제로 두 소자가 병렬인지
이미지를 다시 검토하라.


────────────────────────────────────────────────────────────────────────────
절대 금지
────────────────────────────────────────────────────────────────────────────

* 동일한 도선을 여러 node_id로 분할
* node.connected_elements 와 element.nodes 정보 불일치
* 실제 연결되지 않은 소자를 connected_elements에 포함
* 소자 단자마다 새로운 node_id 생성

노드는 "단자 수"가 아니라
"전기적으로 독립된 전위 영역 수"를 기준으로 생성해야 한다.

────────────────────────────
노드 최소화 규칙
────────────────────────────

노드는 소자 기준이 아니라
전기적 도체 영역 기준이다.

새로운 node_id를 만들기 전에
기존 node_id에 포함될 수 없는지 먼저 검토하라.

회로를 해석한 후
노드 수는
(독립 전위 영역 수)
와 정확히 일치해야 한다.


[target_answer — 최종 타겟 정답 명시]
────────────────────────────────────────────────────────────────────────────
문제 지문을 분석하여 "이 풀이에서 최종적으로 구해야 하는 물리량"을 반드시 명시하라.

- label       : 지문에 등장하는 변수명 그대로 (예: "Vth", "Rth", "In", "Vo", "I_R2")
                지문에서 구하라고 명시된 변수명을 OCR 그대로 사용. 임의 변환 금지.
- value       : 풀이 결과가 이미 알려진 경우 숫자로, 미지 상태이면 null
                (대부분 null로 초기화; 수치 엔진이 나중에 채움)
- unit        : "V" | "A" | "Ω" | "S" | "W" 중 지문에 맞는 단위
- description : 어떤 값인지 한 문장으로 설명
                예: "개방 단자 a-b 사이의 테브난 등가 전압"
                    "독립 전원 비활성화 후 단자 a-b에서 본 등가저항"

★ CALCULATION 문제: 지문이 구하라고 한 소자 전압 또는 전류가 target_answer.
★ CONVERSION 문제: 등가회로의 핵심 파라미터(Vth, Rth, In, Rn 등)가 target_answer.
★ 지문에 "(1) Vth, (2) Rth" 처럼 복수이면 해당 menu_id의 요구사항 하나만 target_answer로 설정.


[step_defs 배열 — 풀이 단계 명세]
- 각 문제 유형에 맞는 정석 풀이 단계를 순서대로 작성하세요.
- CALCULATION 예시 단계:
    1. 노드 설정 및 접지 선택
    2. 종속 전원 제어 변수 수식화 (해당 시)
    3. 노드별 KCL 방정식 수립
    4. 연립방정식 행렬화 및 풀이
    5. 소자 전류 계산 및 방향 정제
    6. 최종 결과 정리
- CONVERSION 예시 단계 (테브난):
    1. 원본 회로 확인
    2. 개방 전압(Voc) 측정 — R_L 제거
    3. 독립 전원 비활성화 (전압원→단락, 전류원→개방)
    4. 등가저항(Rth) 계산
    5. 테브난 등가회로 조립
- description: 수식(LaTeX 포함 가능)과 물리적 의미를 친절하게 서술.
  ★ 수식 LaTeX 아래첨자 필수 규칙 (step_defs description 전용):
    - 변수명에 숫자가 붙는 경우 반드시 LaTeX 아래첨자 문법 사용.
      올바른 예: $n_{3}$, $V_{1}$, $n_{12}$
      금지된 예: n_3, n3, V_1, V1 (렌더러가 깨짐)
    - 분수 표현은 반드시 \dfrac{분자}{분모} 사용. 슬래시(/) 표기 금지.
      올바른 예: $\dfrac{n_{3} - n_{1}}{4000}$
      금지된 예: (n_3 - n_1)/4000, n3/4000
    - 연립방정식은 반드시 $$ ... \begin{cases} ... \end{cases} $$  디스플레이 블록으로 감쌀 것.
    - 인라인 수식은 $ ... $ 로 감쌀 것.
- theory_ids: 사용 가능한 ID: th_node_method, th_kcl, th_ohm, th_dep_source,
                    th_matrix, th_current_dir, th_thevenin, th_norton,
                    th_superposition, th_mesh
- topology_change: 소자 삭제/합성 시 element_id 목록 기입, 변화 없으면 null

────────────────────────────────────────────────────────────────────────────
중요 규칙 요약
────────────────────────────────────────────────────────────────────────────
1. 접지 노드(n0)의 symbol은 반드시 "0". "GND", "n0" 절대 금지
2. node_voltage, current_value 의 초기값은 모두 null
3. terminal_voltages 필드는 항상 빈 객체({})로 초기화하세요. 계산은 백엔드 엔진이 전담합니다.
4. 모든 소자 is_changed: false, current_value: null 로 초기화
5. JSON 이외의 어떠한 텍스트도 출력하지 마세요.
6. parsed_question_text 에 ** 마크다운 기호를 절대 사용하지 마세요.
7. target_answer는 모든 sub_solution에 반드시 포함 — label/unit만이라도 확정하고 value는 null""").strip()


_VISION_USER_PROMPT = (
    "위 지침에 따라 이 회로 이미지를 분석하고 JSON을 출력하세요."
)


# ═════════════════════════════════════════════════════════════════════════════
# Claude Vision 호출
# ═════════════════════════════════════════════════════════════════════════════

def _call_claude_vision(
    image_bytes : bytes,
    media_type  : str = "image/jpeg",
) -> dict:
    """
    이미지를 Claude 3.5 Sonnet Vision API 로 전송하여 회로 구조화 JSON을 반환.

    Parameters
    ----------
    image_bytes : 이미지 원본 바이트
    media_type  : MIME 타입 ("image/jpeg" | "image/png" | "image/webp")

    Returns
    -------
    Claude 가 반환한 JSON 파싱 결과 dict
    """
    client      = _get_client()
    b64_image   = base64.standard_b64encode(image_bytes).decode("utf-8")

    message = client.messages.create(
        model      = "claude-sonnet-4-6",
        max_tokens = 8192,
        system     = _VISION_SYSTEM_PROMPT,
        messages   = [
            {
                "role": "user",
                "content": [
                    {
                        "type"  : "image",
                        "source": {
                            "type"       : "base64",
                            "media_type" : media_type,
                            "data"       : b64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": _VISION_USER_PROMPT,
                    },
                ],
            }
        ],
    )

    raw_text = message.content[0].text.strip()

    # JSON 코드펜스 방어적 제거
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$",          "", raw_text)

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Claude Vision 응답을 JSON으로 파싱하는 데 실패했습니다.\n"
            f"원인: {e}\n"
            f"응답 원문(첫 500자): {raw_text[:500]}"
        )



# ═════════════════════════════════════════════════════════════════════════════
# 단일 sub_solution 파이프라인 실행
# ═════════════════════════════════════════════════════════════════════════════

def _run_sub_solution(
    sub_raw      : dict,
    problem_type : str,
) -> dict:
    initial_circuit: dict      = sub_raw["initial_circuit"]
    ai_target_answer = sub_raw.get("target_answer") or None

    if problem_type != "CALCULATION":
        raise ValueError("현재 버전은 CALCULATION 문제만 지원합니다.")
        
    engine_result = run_full_pipeline(initial_circuit)

    if ai_target_answer:
        engine_result["target_answer"] = ai_target_answer

    # ── UI 출력을 위한 노드 이름(Key) 친절하게 변환 로직 ─────────
    friendly_solution = {}
    target_label = ai_target_answer.get("label", "") if ai_target_answer else ""
    norm_target = target_label.lower().replace("_", "") 
    target_unit = ai_target_answer.get("unit", "").upper() if ai_target_answer else ""

    elements_map = {el["element_id"]: el for el in engine_result["initial_circuit"]["elements"]}

    # 1. 노드 전압들 매핑 (기존 로직)
    for sym, val in engine_result["solution"].items():
        node = next((n for n in engine_result["initial_circuit"]["nodes"] if n.get("symbol") == sym), None)
        
        if not node:
            friendly_solution[sym] = val
            continue
            
        nid = node["node_id"]
        node_label = node.get("label", "")
        norm_sym = sym.lower().replace("_", "")
        norm_label = node_label.lower().replace("_", "")
        
        is_target = False
        # 구하려는 값이 "전압(V)"일 때만 노드에서 타겟 매칭
        if norm_target and target_unit != "A" and (norm_target in norm_sym or norm_target in norm_label):
            is_target = True

        conn_ids = node.get("connected_elements", [])
        desc = "위치 알 수 없음"
        
        if conn_ids:
            first_eid = conn_ids[0]
            el = elements_map.get(first_eid)
            if el:
                if el.get("nodes", {}).get("positive_node_id") == nid:
                    terminal = "+단자"
                elif el.get("nodes", {}).get("negative_node_id") == nid:
                    terminal = "-단자"
                else:
                    terminal = "연결단"
                el_name = el.get("label", el["element_id"])
                desc = f"{el_name}의 {terminal}"

        display_name = node_label if node_label else sym

        if is_target:
            friendly_key = f"★ {target_label} ({desc})"
        else:
            friendly_key = f"{display_name} ({desc})"

        friendly_solution[friendly_key] = val

    # ── [추가] 2. 전류 타겟 매핑 (소자 전류 찾기) ─────────
    # 구하려는 값이 "전류(A)"이거나 이름이 "I"로 시작하면 result_circuit의 elements를 뒤져서 값을 찾습니다.
    if norm_target and (target_unit == "A" or target_label.upper().startswith("I")):
        for el in engine_result["result_circuit"]["elements"]:
            el_label = el.get("label", "")
            norm_el_label = el_label.lower().replace("_", "")
            
            # 소자 라벨에 I_o, I_x 등이 포함되어 있다면 매칭 성공!
            if norm_target in norm_el_label:
                val = el.get("current_value")
                if val is not None:
                    cd = el.get("current_direction", {})
                    frm_nid = cd.get("from_node_id", "?")
                    to_nid = cd.get("to_node_id", "?")
                    
                    # 보기 편하도록 노드 ID 대신 LaTeX 형태나 라벨로 변환 가능
                    friendly_key = f"★ {target_label} ({frm_nid} → {to_nid} 방향)"
                    friendly_solution[friendly_key] = val
                break # 찾았으므로 루프 종료

    return {
        "menu_id"          : sub_raw.get("menu_id", 1),
        "menu_title"       : sub_raw.get("menu_title", "풀이"),
        "initial_circuit"  : engine_result["initial_circuit"],
        "result_circuit"   : engine_result["result_circuit"],
        "solution"         : friendly_solution,
        "steps"            : engine_result["steps"],
        "applied_theories" : engine_result["applied_theories"],
    }


# ═════════════════════════════════════════════════════════════════════════════
# 공개 API — 이미지 → 최종 응답 JSON
# ═════════════════════════════════════════════════════════════════════════════

def analyze_circuit_image(
    image_bytes: bytes,
    media_type : str = "image/jpeg",
) -> dict[str, Any]:
    """
    회로 이미지 바이트를 받아 전체 파이프라인을 실행하고 최종 결과를 반환.

    Parameters
    ----------
    image_bytes : 업로드된 이미지 원본 바이트
    media_type  : MIME 타입 ("image/jpeg" | "image/png" | "image/webp")

    Returns
    -------
    {
        "problem_type"                 : "CONVERSION" | "CALCULATION",
        "parsed_question_text"         : str,
        "sub_solutions"                : [
            {
                "menu_id"          : int,
                "menu_title"       : str,
                "initial_circuit"  : dict,
                "result_circuit"   : dict,
                "solution"         : dict,
                "steps"            : list,
                "applied_theories" : list,
            },
            ...
        ]
    }

    Raises
    ------
    EnvironmentError  ANTHROPIC_API_KEY 미설정
    ValueError        Claude 응답 JSON 파싱 실패 / SymPy 해 없음
    KeyError          회로 데이터 필수 필드 누락
    """
    # ── 1단계: Claude Vision 호출 ────────────────────────────────────────────
    ai_output = _call_claude_vision(image_bytes, media_type)

    problem_type         : str       = ai_output.get("problem_type", "CALCULATION")
    parsed_question_text : str       = ai_output.get("parsed_question_text", "")
    sub_solutions_raw    : list[dict]= ai_output.get("sub_solutions", [])

    if not sub_solutions_raw:
        raise ValueError(
            "Claude Vision 응답에 sub_solutions 배열이 없거나 비어 있습니다. "
            "이미지가 회로도인지 확인하고 다시 업로드해 주세요."
        )

    # ── 2단계: 각 sub_solution 수학 엔진 통과 ───────────────────────────────
    completed_sub_solutions: list[dict] = []
    for sub_raw in sub_solutions_raw:
        result = _run_sub_solution(sub_raw, problem_type)
        completed_sub_solutions.append(result)

    # ── 3단계: 최종 응답 조립 ────────────────────────────────────────────────
    return {
        "problem_type"                : problem_type,
        "parsed_question_text"        : parsed_question_text,
        "sub_solutions"               : completed_sub_solutions
    }
