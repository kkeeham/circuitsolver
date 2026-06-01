"""
main.py
─────────────────────────────────────────────────────────────────────────────
회로 해석 웹 서비스 — FastAPI 서버 진입점  (v3 · 멀티모달 Vision 통합)

엔드포인트 구성
───────────────
  GET  /                헬스체크
  POST /api/solve       텍스트 회로 JSON → 수학 해석 (4단계 기존 기능 유지)
  POST /api/upload-image  회로 이미지 파일 → Vision 분석 → 전체 파이프라인

필수 환경변수
─────────────
  ANTHROPIC_API_KEY     Claude API 키
  (선택) ALLOWED_ORIGINS  쉼표 구분 도메인 목록 (기본값: *)

실행 방법
─────────
  pip install fastapi uvicorn anthropic sympy python-multipart
  python main.py
  → http://localhost:8000/docs  (Swagger UI)
"""

from __future__ import annotations

import asyncio
import os
import traceback
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, List, Optional

import anthropic
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── 내부 모듈 임포트 ─────────────────────────────────────────────────────────
from circuit_engine_v2 import run_full_pipeline
from ai_pipeline import analyze_circuit_image


# ═════════════════════════════════════════════════════════════════════════════
# 응답 Pydantic 스키마
# ═════════════════════════════════════════════════════════════════════════════

class SubSolutionResponse(BaseModel):
    """단일 sub_solution (풀이 메뉴 분기) 응답 스키마."""
    menu_id:           int            = Field(..., description="메뉴 순번 (1-based)")
    menu_title:        str            = Field(..., description="메뉴 제목 (지문 요구사항 그대로)")
    solution:          dict           = Field(..., description="전압·전류 수치 해 { 변수명: 값 }")
    steps:             List[Any]      = Field(..., description="단계별 풀이 텍스트 배열")
    applied_theories:  List[Any]      = Field(..., description="적용된 이론 목록")


class CircuitImageResponse(BaseModel):
    """POST /api/analyze 최종 응답 스키마 — 해석 JSON만 반환 (SVG/회로도 필드 없음)."""
    problem_type:         str                      = Field(..., description="문제 유형: 'CONVERSION' | 'CALCULATION'")
    parsed_question_text: str                      = Field(..., description="OCR 추출 문제 지문 (**bold** 마크업 포함)")
    sub_solutions:        List[SubSolutionResponse] = Field(..., description="풀이 메뉴 분기 배열")


# ═════════════════════════════════════════════════════════════════════════════
# 허용 MIME 타입 및 파일 크기 제한
# ═════════════════════════════════════════════════════════════════════════════

_ALLOWED_MIME: dict[str, str] = {
    "image/jpeg" : "image/jpeg",
    "image/jpg"  : "image/jpeg",
    "image/png"  : "image/png",
    "image/webp" : "image/webp",
    "image/gif"  : "image/gif",
}
_MAX_IMAGE_BYTES = 20 * 1024 * 1024   # 20 MB


# ═════════════════════════════════════════════════════════════════════════════
# 애플리케이션 수명주기 (lifespan)
# ═════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 환경변수 사전 검증"""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "\n⚠️  경고: ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.\n"
            "   POST /api/upload-image 엔드포인트는 동작하지 않습니다.\n"
            "   .env 파일에 ANTHROPIC_API_KEY=sk-ant-... 를 추가하거나\n"
            "   터미널에서 export ANTHROPIC_API_KEY=<키> 를 실행하세요.\n"
        )
    else:
        print("✅  ANTHROPIC_API_KEY 확인 완료.")
    yield


# ═════════════════════════════════════════════════════════════════════════════
# FastAPI 인스턴스
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "Circuit Solver API",
    description = (
        "회로 이미지 업로드 → Claude Vision 독해 → 마디 해석법 자동 풀이 서비스.\n\n"
        "**주요 엔드포인트**\n"
        "- `POST /api/upload-image` — 회로 이미지 파일 한 장으로 전체 파이프라인 실행\n"
        "- `POST /api/solve` — initial_circuit JSON 직접 전송 (텍스트 경로)"
    ),
    version     = "3.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)


# ═════════════════════════════════════════════════════════════════════════════
# CORS 미들웨어
# ═════════════════════════════════════════════════════════════════════════════

_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
_origins     = [o.strip() for o in _origins_env.split(",")] if _origins_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins     = _origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ═════════════════════════════════════════════════════════════════════════════
# 전역 예외 핸들러
# ═════════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    tb = traceback.format_exc()
    print(f"[UNHANDLED EXCEPTION] {request.url}\n{tb}")
    return JSONResponse(
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        content     = {"detail": "서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."},
    )


# ═════════════════════════════════════════════════════════════════════════════
# 공통 내부 헬퍼 — initial_circuit 사전 검증 + 파이프라인 안전 실행
# ═════════════════════════════════════════════════════════════════════════════

def _validate_initial_circuit(circuit: dict) -> None:
    """필수 필드 검증. 문제 있으면 HTTPException(400) 발생."""
    for key in ("elements", "nodes"):
        if key not in circuit:
            nd["terminal_voltages"] = {}
    if not circuit.get("elements"):
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "elements 배열이 비어 있습니다. 소자를 하나 이상 포함해야 합니다.",
        )
    if not circuit.get("nodes"):
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "nodes 배열이 비어 있습니다. 노드를 하나 이상 포함해야 합니다.",
        )
    if not any(nd.get("symbol") == "0" for nd in circuit["nodes"]):
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = (
                "접지 노드(symbol: '0')가 없습니다. "
                "노드 중 하나의 symbol 값을 '0'으로 설정하여 접지를 지정해 주세요."
            ),
        )

    # ── terminal_voltages 형식 검증 ─────────────────────────────────────────
    # circuit_engine_v2 의 apply_solution_to_circuit 은 terminal_voltages 가
    # 반드시 dict 임을 가정한다 (list 보정 코드 삭제됨).
    # LLM 또는 클라이언트가 list 로 넘기는 경우를 서버 진입점에서 차단한다.
    for idx, nd in enumerate(circuit["nodes"]):
        tv = nd.get("terminal_voltages")
        if tv is not None and not isinstance(tv, dict):
            raise HTTPException(
                status_code = status.HTTP_400_BAD_REQUEST,
                detail      = (
                    f"nodes[{idx}] 의 terminal_voltages 가 dict 가 아닙니다 "
                    f"(받은 타입: {type(tv).__name__}). "
                    "terminal_voltages 는 반드시 {{ \"소자ID\": {{...}} }} 형태의 JSON 객체여야 합니다."
                ),
            )

    # ── 각 노드에 symbol 필드 존재 여부 검증 ────────────────────────────────
    # apply_solution_to_circuit 에서 symbol 누락 시 KeyError 로 즉시 서버 다운.
    # 진입점에서 미리 검사하여 명확한 에러 메시지를 반환한다.
    for idx, nd in enumerate(circuit["nodes"]):
        if "symbol" not in nd:
            raise HTTPException(
                status_code = status.HTTP_400_BAD_REQUEST,
                detail      = (
                    f"nodes[{idx}] 에 필수 필드 'symbol' 이 없습니다. "
                    "모든 노드에 symbol 값(예: 'V1', '0')이 있어야 합니다."
                ),
            )


async def _safe_run_pipeline(initial_circuit: dict) -> dict:
    """run_full_pipeline 을 스레드 풀에서 실행 + 엔진 예외 → HTTP 400 변환.

    run_full_pipeline 은 SymPy 기호 연산(CPU 바운드) + 순수 Python 루프로 구성된
    동기 블로킹 함수이므로, 이벤트 루프를 점유하지 않도록 run_in_executor 로 감싼다.

    circuit_engine_v2 변경 사항 반영 (1차 수정 5/31):
    - _node_voltage_map : node_voltage 가 None 이면 0V 로 넘기지 않고 ValueError 발생
    - calculate_currents_and_directions : Vp/Vn 미계산 시 ValueError 발생
    - _build_kcl_equations : 제어소자(CCCS/VCCS) 관련 오류 시 ValueError/KeyError 발생
    위 경우 모두 아래 except 블록에서 포착되어 HTTP 400 으로 변환된다.
    """
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, partial(run_full_pipeline, initial_circuit))
    except ValueError as ve:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = (
                "회로 연산에 실패했습니다. "
                "노드 전압 미계산, 소자 연결 상태, 종속 전원 설정을 다시 확인해 주세요. "
                f"(원인: {ve})"
            ),
        )
    except KeyError as ke:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = (
                "회로 데이터 구조에 오류가 있습니다. "
                "필수 필드 누락, 소자 ID 참조 오류, 또는 symbol 누락일 수 있습니다. "
                f"(누락 키: {ke})"
            ),
        )
    except ZeroDivisionError:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = (
                "0 Ω 저항 또는 0으로 나누는 연산이 발생했습니다. "
                "저항 소자의 value 값이 0 이 아닌지 확인해 주세요."
            ),
        )
    except Exception as exc:
        # run_full_pipeline 은 내부에서 Exception 을 catch 해 fallback dict 를 반환하므로,
        # 이 블록은 실질적으로 도달하지 않는다. 예상치 못한 외부 오류에 대한 안전망으로 유지.
        print(f"[ENGINE ERROR]\n{traceback.format_exc()}")
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = (
                "회로 연산에 실패했습니다. 소자 연결 상태 및 종속 전원 설정을 다시 확인해 주세요. "
                f"(상세: {type(exc).__name__}: {exc})"
            ),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 헬스체크
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/", summary="헬스체크", tags=["Health"])
async def health_check() -> dict[str, str]:
    return {
        "status" : "ok",
        "version": "3.0.0",
        "message": "서버가 정상 동작 중입니다.",
    }


# ═════════════════════════════════════════════════════════════════════════════
# [기존] POST /api/solve — initial_circuit JSON 직접 수신
# ═════════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/solve",
    summary     = "회로 JSON 직접 해석",
    description = (
        "`initial_circuit` JSON 을 Body 로 전송하면 A → B → C 파이프라인을 실행합니다.\n\n"
        "```json\n{ \"initial_circuit\": { ... } }\n```"
    ),
    tags        = ["Circuit Solver"],
    status_code = status.HTTP_200_OK,
)
async def solve_circuit(payload: dict[str, Any]) -> dict[str, Any]:
    if "initial_circuit" in payload:
        initial_circuit: dict = payload["initial_circuit"]
    elif "elements" in payload and "nodes" in payload:
        initial_circuit = payload
    else:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = (
                "요청 Body 에 'initial_circuit' 키가 없습니다. "
                "{ \"initial_circuit\": { ... } } 형태로 전송해 주세요."
            ),
        )
    _validate_initial_circuit(initial_circuit)
    return await _safe_run_pipeline(initial_circuit)


# ═════════════════════════════════════════════════════════════════════════════
# ai_pipeline 딕셔너리 → Pydantic 응답 모델 타입 정규화 헬퍼
# ═════════════════════════════════════════════════════════════════════════════

def _normalize_and_build_response(result: dict) -> CircuitImageResponse:
    """
    analyze_circuit_image() 가 반환하는 원시 딕셔너리를 CircuitImageResponse
    Pydantic 모델로 안전하게 변환한다.

    변환 대상 필드
    ──────────────
    sub_solutions : List[dict]  →  List[SubSolutionResponse]
        각 원소에 menu_id·menu_title·solution·steps·applied_theories 가
        있는지 확인하며 변환. 누락 키는 HTTPException(422) 으로 즉시 반환.
    """

    # ── sub_solutions 정규화 ─────────────────────────────────────────────────
    raw_sub_solutions: list = result.get("sub_solutions", [])
    if not isinstance(raw_sub_solutions, list):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "ai_pipeline 이 반환한 sub_solutions 가 배열이 아닙니다. "
                f"(받은 타입: {type(raw_sub_solutions).__name__})"
            ),
        )

    normalized_sub_solutions: List[SubSolutionResponse] = []
    for idx, sub in enumerate(raw_sub_solutions):
        if not isinstance(sub, dict):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"sub_solutions[{idx}] 가 dict 가 아닙니다. "
                    f"(받은 타입: {type(sub).__name__})"
                ),
            )
        try:
            normalized_sub_solutions.append(
                SubSolutionResponse(
                    menu_id          = int(sub["menu_id"]),
                    menu_title       = str(sub["menu_title"]),
                    solution         = dict(sub["solution"]),
                    steps            = list(sub["steps"]),
                    applied_theories = list(sub["applied_theories"]),
                )
            )
        except KeyError as missing_key:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"sub_solutions[{idx}] 에 필수 필드 {missing_key} 가 없습니다. "
                    "ai_pipeline 출력과 SubSolutionResponse 스키마를 확인하세요."
                ),
            )
        except (TypeError, ValueError) as cast_err:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"sub_solutions[{idx}] 의 필드 타입 변환에 실패했습니다. "
                    f"원인: {cast_err}"
                ),
            )

    # ── 최종 CircuitImageResponse 조립 ───────────────────────────────────────
    try:
        return CircuitImageResponse(
            problem_type         = result["problem_type"],
            parsed_question_text = result["parsed_question_text"],
            sub_solutions        = normalized_sub_solutions,
        )
    except KeyError as missing_key:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"ai_pipeline 응답에 최상위 필수 필드 {missing_key} 가 없습니다. "
                "CircuitImageResponse 스키마와 ai_pipeline 출력을 확인하세요."
            ),
        )


# ═════════════════════════════════════════════════════════════════════════════
# POST /api/analyze — 이미지 → Vision → 해석 JSON 반환 (SVG/회로도 없음)
# ═════════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/upload-image",
    summary     = "회로 이미지 업로드 (프론트엔드 호환 alias)",
    description = "`/api/analyze` 와 동일한 동작. 프론트엔드 호환을 위해 유지.",
    tags        = ["Vision Pipeline"],
    status_code = status.HTTP_200_OK,
)
async def upload_image_alias(
    file: UploadFile = File(..., description="분석할 회로 이미지 파일 (JPEG / PNG / WebP / GIF, 최대 20 MB)"),
) -> CircuitImageResponse:
    return await upload_image(file)


@app.post(
    "/api/analyze",
    summary     = "회로 이미지 업로드 → 자동 분석 및 풀이",
    description = (
        "**회로와 문제 지문이 포함된 이미지 파일 한 장을 업로드하면**\n\n"
        "1. Claude Vision 이 회로 구조와 문제 지문을 독해\n"
        "2. 문제 유형(`CONVERSION` / `CALCULATION`) 자동 판정\n"
        "3. 지문의 요구사항 수만큼 `sub_solutions[]` 메뉴 분기 생성\n"
        "4. 각 sub_solution 을 수학 엔진에 통과시켜 전압·전류 연산 완료\n\n"
        "**지원 형식:** JPEG, PNG, WebP, GIF  /  **최대 크기:** 20 MB\n\n"
        "**multipart/form-data** 로 `file` 필드에 이미지를 첨부하세요.\n\n"
        "**응답 구조:**\n"
        "```json\n"
        "{\n"
        "  \"problem_type\": \"CONVERSION\",\n"
        "  \"parsed_question_text\": \"**문제 내용...**\",\n"
        "  \"sub_solutions\": [\n"
        "    {\n"
        "      \"menu_id\": 1,\n"
        "      \"menu_title\": \"(1) 테브난 등가회로 변환\",\n"
        "      \"solution\": { \"V1\": 20.0 },\n"
        "      \"steps\": [ \"...\" ],\n"
        "      \"applied_theories\": [ \"...\" ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "```"
    ),
    tags        = ["Vision Pipeline"],
    status_code = status.HTTP_200_OK,
)
async def upload_image(
    file: UploadFile = File(
        ...,
        description = "분석할 회로 이미지 파일 (JPEG / PNG / WebP / GIF, 최대 20 MB)",
    ),
) -> CircuitImageResponse:
    # ── API 키 사전 확인 ─────────────────────────────────────────────────────
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = (
                "ANTHROPIC_API_KEY 환경변수가 설정되지 않아 Vision 서비스를 사용할 수 없습니다. "
                "서버 관리자에게 문의하거나 .env 파일에 API 키를 등록해 주세요."
            ),
        )

    # ── MIME 타입 검증 ───────────────────────────────────────────────────────
    content_type = (file.content_type or "").lower()
    if content_type not in _ALLOWED_MIME:
        raise HTTPException(
            status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail      = (
                f"지원하지 않는 파일 형식입니다: '{content_type}'. "
                "JPEG, PNG, WebP, GIF 형식의 이미지만 업로드할 수 있습니다."
            ),
        )
    media_type = _ALLOWED_MIME[content_type]

    # ── 파일 크기 검증 ───────────────────────────────────────────────────────
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "업로드된 파일이 비어 있습니다.",
        )
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        size_mb = len(image_bytes) / (1024 * 1024)
        raise HTTPException(
            status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail      = (
                f"파일 크기({size_mb:.1f} MB)가 최대 허용 크기(20 MB)를 초과합니다. "
                "이미지를 압축한 후 다시 업로드해 주세요."
            ),
        )

    # ── Vision 파이프라인 실행 ───────────────────────────────────────────────
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, partial(analyze_circuit_image, image_bytes, media_type))

    except EnvironmentError as ee:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(ee))

    except ValueError as ve:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = (
                "이미지 분석 또는 회로 연산에 실패했습니다. "
                "회로도가 명확하게 찍힌 이미지인지 확인하고 다시 시도해 주세요. "
                f"(원인: {ve})"
            ),
        )

    except KeyError as ke:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = (
                "AI 가 추출한 회로 데이터에 필수 필드가 없습니다. "
                f"(누락 키: {ke}) "
                "이미지를 다시 업로드하거나 회로가 더 잘 보이는 사진으로 시도해 주세요."
            ),
        )

    except anthropic.APIConnectionError:
        raise HTTPException(
            status_code = status.HTTP_502_BAD_GATEWAY,
            detail      = (
                "Anthropic API 서버에 연결할 수 없습니다. "
                "네트워크 상태를 확인하거나 잠시 후 다시 시도해 주세요."
            ),
        )

    except anthropic.RateLimitError:
        raise HTTPException(
            status_code = status.HTTP_429_TOO_MANY_REQUESTS,
            detail      = (
                "API 요청 한도를 초과했습니다. "
                "잠시 후 다시 시도하거나 Anthropic 콘솔에서 사용량을 확인해 주세요."
            ),
        )

    except anthropic.APIStatusError as ase:
        raise HTTPException(
            status_code = status.HTTP_502_BAD_GATEWAY,
            detail      = f"Anthropic API 오류가 발생했습니다. (HTTP {ase.status_code}: {ase.message})",
        )

    except Exception as exc:
        print(f"[VISION PIPELINE ERROR]\n{traceback.format_exc()}")
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = (
                "이미지 분석 파이프라인에서 예상치 못한 오류가 발생했습니다. "
                f"({type(exc).__name__}: {exc})"
            ),
        )

    # sub_solutions 의 각 원소를 SubSolutionResponse 로 변환하여 반환한다.
    # Pydantic v2 는 List[dict] → List[SubSolutionResponse] 를 자동 coerce 하지 않으므로
    # _normalize_and_build_response 에서 명시적으로 변환한다.
    return _normalize_and_build_response(result)


# ═════════════════════════════════════════════════════════════════════════════
# uvicorn 자동 실행 진입점
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host   = "0.0.0.0",
        port   = 8000,
        reload = True,
    )
