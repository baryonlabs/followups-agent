# 제안요청서 요구사항 → followups-agent 기반 "구조적 멀티턴" 설계

제안요청서의 8개 요구(멀티모달 · 멀티턴 · 멀티에이전트 · 멀티목표 · 멀티스키마 Text-to-SQL · 멀티가이드라인 · 정책 · 사용자 메모리)와 공공 보안 기준을 이 저장소의 구조 위에 얹는 안입니다.
핵심은 하나입니다. **대화 이력을 요약해서 넘기지 않고, 대화 상태를 타입이 있는 객체(작업 스택 · 슬롯 · 엔티티 · 칩 payload)로 들고 간다.** 이 문서에서는 이것을 "구조적 멀티턴"이라고 부릅니다.

---

## 1. 왜 "요약 주입"이 아니라 "구조적 멀티턴"인가

참고안(중앙 제어형 멀티 에이전트)은 "라우터가 과거 맥락을 요약해 하위 에이전트에 주입"하는 방식입니다. 이 방식은 턴이 쌓일수록 세 가지가 깨집니다.

| 문제 | 요약 주입 | 구조적 멀티턴 (이 저장소) |
|---|---|---|
| 어디까지 했는지 | 요약문에 "결제 준비함"이라고 적혀 있을 뿐, 실제로 끝났는지 모름 | `Task.status` (`pending/blocked/awaiting_confirm/done/failed/cancelled`) |
| 무엇이 비었는지 | LLM이 매번 다시 판단 | `Task.missing` = 카탈로그 필수 슬롯 − 채워진 슬롯 (규칙, `compute_missing`) |
| "아까 그거" | 요약에서 사라지면 해석 불가 | `entity_memory.last_product` 등 id로 보관 (`new_task`의 지시어 보강) |
| 사용자의 선택 | 자연어로 다시 들어와 다시 추출 (오인식 가능) | 칩 payload `{op, task_id, slots}`가 의도 추출을 건너뜀 (`apply_payload`) |
| 되돌릴 수 없는 실행 | 프롬프트로 "확인받아라" | `interrupt` + 체크포인터, `idempotency_key` |
| 검증 | 대화를 사람이 읽어야 함 | 최종 상태를 코드로 단정 (`run.py`의 시나리오 검증, `tests/test_rules.py`) |

LLM은 **발화 → 구조화**(의도 추출)와 **구조 → 문장**(답변 작성)의 양 끝에만 쓰고, 그 사이의 상태 전이(병합 · 의존성 · 실행 순서 · 후속 질문)는 순수 함수(`state.py`)로 둡니다. 공공 사업에서 요구하는 "설명 가능성 · 재현성 · 감사 추적"이 여기서 나옵니다.

---

## 2. 요구사항 ↔ 현재 구현 ↔ 확장 매핑

| 요구사항 | 이미 있는 것 | 확장할 것 |
|---|---|---|
| **멀티턴** | `State`(messages · task_stack · entity_memory) + LangGraph 체크포인터, 최근 6메시지만 LLM에 전달 | `InMemorySaver` → `PostgresSaver`, 세션 TTL, 첨부·SQL 결과도 엔티티로 |
| **멀티목표** | 한 발화 → 여러 `Task`, `depends_on`, `plan()`이 실행 가능 작업 선정, `Send` 병렬 | 의존성 규칙을 카탈로그 선언으로(`produces/consumes`), 보고서 같은 합성 목표 |
| **멀티에이전트** | Router(의도 추출) + workers + supervisor(subagents as tools) + 빠른 경로 | `sql_agent`, `attachment_reader`, `report_writer` 추가 |
| **멀티모달** | 없음 | 입구에서 이미지 → 구조화 사실(엔티티)로 한 번만 변환 |
| **멀티스키마 Text-to-SQL** | 없음 (MCP 도구 동적 바인딩 `Toolbox.subset`은 있음) | 2단계 스키마 라우팅을 **슬롯 파이프라인**으로, 모호성은 칩으로 |
| **멀티가이드라인** | 프롬프트에 하드코딩 | 가이드라인 레지스트리, 작업 단위로 id를 붙여 조립 |
| **정책** | 위험도(`risk`) → 결제 확인(HITL), 멱등 키 | 입력·출력 가드 노드, SQL 검증기, PII 컬럼 차단, 정책 칩 |
| **사용자 메모리** | 세션 안 `entity_memory`만 | 장기 프로필(Postgres), 턴 종료 후 비동기 추출, **사용자 확인 칩으로만 기록** |

---

## 3. 확장된 턴 흐름

```
input_guard ──(차단)──────────────────────────────────────────▶ policy_reply ─▶ END
    │
begin_turn  (profile 로드, turn+1)
    │
    ├─ 칩 payload ─────────▶ apply_chip ───────────────┐
    │                                                  │
    └─ 자연어 ─▶ attachment_reader? ─▶ intent_extractor ─▶ stack_merger
                 (이미지 있을 때만)                       │
                                                       ▼
                         planner ⇄ workers (Send 병렬)
                           ├ simple_worker       (기존)
                           ├ confirm_gate        (기존, interrupt)
                           ├ supervisor_agent    (기존, 판단형)
                           ├ sql_agent           (신규: 스키마 선택 → 생성 → 검증 → 실행)
                           └ report_writer       (신규: 의존 작업 결과 + 가이드라인으로 합성)
                                                       │
                         aggregator ∥ followup_generator
                                                       │
                                   output_guard ─▶ END
                                                       │
                              (비동기) memory_extractor ─▶ 메모리 후보 → 다음 턴 칩
```

바뀌는 점은 세 가지뿐입니다. 앞(`input_guard`)과 뒤(`output_guard`)에 가드가 붙고, worker 종류가 늘고, 턴 밖에서 메모리 추출이 돕니다. planner ⇄ worker 루프, 칩 우선순위, interrupt는 그대로 씁니다.

> 참고안의 "DAG 기반 LangGraph"는 정정이 필요합니다. 이 저장소는 `planner ⇄ workers` **순환**으로 같은 턴 안에서 풀린 의존성을 이어 처리합니다(`MAX_LOOPS=6`으로 상한).

---

## 4. 상태 스키마 확장

```python
IntentName = Literal[
    # 기존 커머스 의도 ...
    "data_query",        # Text-to-SQL
    "attachment_analysis",
    "report_compose",
    "set_preference",    # "앞으로는 ~해줘"
    "other",
]

CATALOG |= {
    "data_query":          {"required": ["metric", "source"], "risk": "low",  "route": "sql",
                            "produces": ["dataset"]},
    "attachment_analysis": {"required": ["attachment_id"],    "risk": "low",  "route": "simple",
                            "produces": ["attachment_facts"]},
    "report_compose":      {"required": ["tone"],             "risk": "low",  "route": "report",
                            "consumes": ["dataset", "attachment_facts"]},
    "set_preference":      {"required": ["rule"],             "risk": "high", "route": "memory"},
}

class Task(TypedDict, total=False):
    ...                       # 기존 필드 그대로
    guidelines: list[str]     # 이 작업에 붙은 가이드라인 id (planner가 부착)
    sql: dict                 # {"tables": [...], "query": "...", "params": {...}, "rows": N}
    result_ref: str           # 대용량 결과는 상태에 넣지 않고 참조 키만

class State(TypedDict, total=False):
    ...                       # 기존 필드 그대로
    attachments: Annotated[dict, merge_dict]  # attachment_id -> {kind, facts, object_key}
    profile: dict             # 장기 메모리 스냅샷 (begin_turn에서 로드, 읽기 전용)
    policy_flags: list[dict]  # input/output guard 판정 기록 (감사용)
```

의존성은 지금 `link_dependencies`에 "주소 변경 → 결제"로 하드코딩되어 있습니다. 목표가 늘면 카탈로그의 `produces/consumes`로 일반화합니다. `report_compose`는 같은 턴에 열린 `dataset`·`attachment_facts` 생산 작업에 자동으로 `depends_on`이 걸리고, 그것들이 `done`이 되는 순간 `plan()`이 실행 대상으로 올립니다(지금의 결제 대기와 같은 원리).

칩 op도 두 개만 늘립니다.

| op | 의미 | 예 |
|---|---|---|
| `fill_slots` (기존) | 빈 슬롯 채우기 | 스키마 후보 중 하나 선택 |
| `skip_task` / `retry` (기존) | 취소 · 재시도 | SQL 시간 초과 후 재시도 |
| `refine` (신규) | 끝난 작업의 슬롯 일부만 바꿔 다시 실행 | "구별로도 나눠줘" → `group_by += 구` |
| `remember` (신규) | 메모리 후보를 장기 프로필에 기록 / 거절 | "앞으로도 적용" |

---

## 5. 요구사항별 설계

### 5.1 멀티모달: 이미지는 입구에서 한 번만 구조화

- `attachment_reader` 노드가 이미지가 있는 턴에만 돌고, VLM으로 **사실 JSON**을 뽑습니다. 예: `{kind: "chart", region: "마포구", period: "2026-01~06", series: [...], anomaly: "2026-04 급감", y_unit: "백만원"}`.
- 원본은 오브젝트 스토리지(MinIO, Langfuse 스택에 이미 있음)에 두고 상태에는 `attachment_id`와 사실만 넣습니다. 이후 턴에 이미지를 다시 모델에 보내지 않습니다.
- "아까 보낸 차트 기준으로"는 `entity_memory.last_attachment`로 해석합니다. 지금 "아까 그 후드티"를 `last_product`로 푸는 것과 같은 방식입니다.
- 추출된 `region`, `period`는 그대로 `data_query`의 슬롯 후보가 됩니다. 이미지와 DB를 잇는 지점이 프롬프트가 아니라 슬롯입니다.
- 사실 추출이 불확실하면(축 단위를 못 읽음 등) `missing`에 넣고 A 칩으로 묻습니다. 예: "단위: 백만원 / 천원".

### 5.2 멀티턴: 체크포인터가 단기 메모리

- 단기 메모리는 별도 Redis 대화 로그가 아니라 **LangGraph 체크포인트 = 스레드 상태**입니다. 결제 확인처럼 `interrupt`로 멈춘 턴이 재개되려면 상태 전체가 보존되어야 하므로, "30분 미활동 시 파기"는 체크포인트 TTL(예: 24시간)과 화면 세션 만료(30분)를 나눠서 정합니다.
- LLM에는 `messages[-6:]` + 열린 작업 + 최근 엔티티만 보냅니다(지금과 같음). 대화가 길어져도 프롬프트 크기가 커지지 않습니다.
- 칩 클릭은 `payload`로 들어와 의도 추출을 건너뜁니다. 공공 서비스에서 많은 턴이 "선택"이므로 LLM 호출 수와 오인식이 함께 줄어듭니다.

### 5.3 멀티목표 · 멀티에이전트: 작업 스택이 목표 트리

참고안의 "목표 트리"는 이미 `task_stack`입니다. 한 발화 "이 차트 원인 찾고, DB에서 업종별 점포 수 뽑아서 보고서 톤으로 정리해줘"는 다음이 됩니다.

```
T1 attachment_analysis  done (입구에서 처리)
T2 data_query           blocked  missing=[source]        ← 스키마 후보 2개
T3 report_compose       blocked  depends_on=[T1, T2]
```

에이전트는 라우터 하나가 다 하지 않고, 처리 경로(`route`)별로 나눕니다.

| 참고안의 에이전트 | 이 저장소에서 |
|---|---|
| 라우터 & 오케스트레이터 | `intent_extractor`(LLM) + `stack_merger` · `planner` · `fan_out`(규칙) |
| 멀티 스키마 Text-to-SQL | `sql_agent` (신규, 5.4) |
| 멀티모달 콘텍스트 생성기 | `attachment_reader`(입구) + `report_writer`/`aggregator`(출구)로 분리 |
| (참고안에 없음) 판단형 상담 | `supervisor_agent` — subagents as tools, 고정 패턴은 빠른 경로 |

오케스트레이터를 LLM 하나로 두지 않는 이유는 지연과 재현성입니다. 이 저장소의 refund 사례에서 supervisor가 "누구를 부를지" 판단하는 왕복을 없앤 빠른 경로가 31.6초 → 7.6~8.4초였습니다.

### 5.4 멀티스키마 Text-to-SQL: 2단계 라우팅을 슬롯 파이프라인으로

`sql_agent`는 하나의 큰 프롬프트가 아니라 단계마다 슬롯을 채우는 작업입니다.

| 단계 | 하는 일 | 막히면 |
|---|---|---|
| ① 스키마 선택 | 스키마 인덱스(테이블 정의서 설명 + 컬럼 설명, pgvector)에서 후보 테이블 검색 | 후보가 여러 DB에 걸치고 점수가 비슷하면 `missing=[source]` → **A 칩**: "상권 DB(월별) / 사업자 DB(분기별)" |
| ② 조건 확정 | `metric`, `period`, `region`, `group_by`를 슬롯으로. 첨부 사실·프로필에서 기본값 | 기간이 없으면 A 칩: "최근 6개월 / 올해 / 직접 입력" |
| ③ SQL 생성 | 선택된 테이블 DDL만 넣어 생성 (코드 특화 모델) | — |
| ④ 검증 | `sqlglot` 파싱: 단일 `SELECT`만, 허용 테이블 목록, PII 컬럼 차단 목록, `LIMIT` 강제, `EXPLAIN` 비용 상한 | 비용 초과 → **B 칩**(확인): "약 1,200만 행을 읽습니다. 실행할까요?" |
| ⑤ 실행 | 읽기 전용 계정, 문장 단위 timeout, 결과는 `result_ref`로 저장 | 실패 → **E 칩**: "다시 시도 / 조건 줄이기" |

- DB는 MCP 서버 단위로 붙입니다(`mcp_servers/`에 `sangkwon_db`, `bizreg_db` 등). 지금 `Toolbox.subset`이 서버별 도구를 골라 바인딩하므로, 스키마 선택 결과에 따라 해당 DB 서버 도구만 노출합니다.
- 후속 턴 "구별로도 나눠줘"는 새 질의가 아니라 `refine`입니다. `T2.slots.group_by`에 `구`를 더하고 ③부터 다시 돕니다. 스키마 선택을 반복하지 않으므로 빠르고, 앞 결과와 같은 테이블을 쓴다는 것이 보장됩니다.
- 검증기와 위험도 판정은 규칙이라 `tests/test_rules.py`처럼 LLM 없이 테스트합니다.

### 5.5 멀티가이드라인: 전역 프롬프트가 아니라 작업에 붙는 id

```python
GUIDELINES = {
    "org.security":     {"scope": "*",               "priority": 0, "text": "..."},  # 항상, 덮어쓸 수 없음
    "sql.readonly":     {"scope": "route:sql",       "priority": 1, "text": "..."},
    "report.q3_tone":   {"scope": "intent:report_compose", "priority": 2, "text": "..."},
    "user.decimal2":    {"scope": "user",            "priority": 3, "text": "소수점 둘째 자리까지"},  # 프로필에서
}
```

- planner가 작업을 실행 대상으로 올릴 때 `scope`가 맞는 id만 `Task.guidelines`에 붙이고, worker는 그 id의 본문만 조립합니다. SQL 에이전트에 보고서 톤 지침이 들어가지 않습니다.
- 우선순위: 기관 정책 > 업무 가이드라인 > 사용자 선호. 사용자 선호는 형식(표, 소수점, 단위)만 바꿀 수 있고 정책을 넘지 못합니다.
- 어떤 가이드라인이 어떤 답에 쓰였는지 `Task.guidelines`가 그대로 남아 트레이스에서 확인됩니다.

### 5.6 정책: 샌드위치 + 실행 지점 통제

| 위치 | 통제 | 구현 |
|---|---|---|
| 입력 | 주민등록번호·전화·계좌 등 PII 패턴, 프롬프트 인젝션 | `input_guard` 노드: 정규식 + 경량 분류기. 차단 시 이유와 **대안 칩** ("대표자 수만 집계하기") |
| 작업 | 위험 작업은 사용자 확인 | 기존 `CATALOG.risk` → `confirm_gate`(interrupt) |
| SQL | 권한 밖 테이블·PII 컬럼·DDL/DML | 5.4 ④ 검증기 + DB 계정 권한(이중) |
| 출력 | 결과에 PII가 섞였는지, 근거 없는 수치 | `output_guard`: 마스킹, `aggregator`의 "done이 아닌 일은 했다고 말하지 않는다" 규칙 유지 |
| 기록 | 판정 근거 | `State.policy_flags` + Langfuse 트레이스 |

"보안을 지키세요"를 프롬프트에 적는 것으로 끝내지 않고, 각 통제가 코드 경로에 있으므로 우회 시나리오를 E2E로 검증할 수 있습니다.

### 5.7 사용자 메모리: 추출은 비동기, 기록은 사용자 확인 후

- 장기 프로필: Postgres `user_profile(user_id, key, value, source_thread, source_turn, created_at, confirmed_at)`. 벡터 DB 없이 키-값 목록으로 시작합니다.
- 턴이 끝나면 `memory_extractor`가 백그라운드(FastAPI BackgroundTasks 또는 큐 워커)로 돌아 "영구 선호·규칙" 후보만 뽑습니다. 출력은 문장이 아니라 `{key, value, evidence_span}`입니다.
- **바로 쓰지 않습니다.** 다음 턴 칩으로 묻습니다: "앞으로도 소수점 둘째 자리까지 표시할까요?" → `remember` payload. 공공 서비스에서 개인정보 수집·이용 동의와 정정·삭제 요구에 대응하려면 사용자가 본 적 없는 기억이 쌓이면 안 됩니다.
- 사용자가 직접 "앞으로는 ~"이라고 말하면 `set_preference` 의도(위험도 high → 확인 칩)로 같은 경로를 탑니다.
- `begin_turn`에서 확정된 프로필만 `State.profile`로 로드하고 5.5의 `user.*` 가이드라인으로 변환합니다.

---

## 6. 시나리오로 본 상태 전이 (공공 데이터 예시)

| 턴 | 사용자 | 상태 변화 | 답 + 칩 |
|---|---|---|---|
| 1 | [차트 이미지] "4월에 왜 떨어졌는지, 같은 기간 업종별 점포 수 DB에서 뽑아서 3분기 보고서 톤으로 정리해줘" | `attachment_reader` → T1 done(region=마포구, period=2026-01~06). T2 data_query blocked `missing=[source]`. T3 report_compose blocked `depends_on=[T1,T2]` | "차트는 마포구 1~6월이고 4월 급감이 보입니다. 점포 수는 두 DB에 있습니다." 칩: **상권 DB(월별)** / 사업자 DB(분기별) |
| 2 | (칩 클릭) | `apply_chip` → T2 pending → SQL 생성·검증·실행 → done → T3 의존성 해제 → 같은 턴에 report_writer 실행 | 보고서 톤 요약. 칩(D): 구별로 나누기 / 전년 동기 비교 |
| 3 | "구별로도 나눠줘" | T2 `refine` group_by+=구 (스키마 선택 생략) → T3 재합성 | 구별 표 |
| 4 | "앞으로 숫자는 반올림하지 마" | `set_preference` → 확인 대기 | 칩(B): **앞으로도 적용** / 이번만 |
| 5 | "대표자 주민번호랑 같이 명단 뽑아줘" | `input_guard` 차단 + `policy_flags` 기록, 작업 생성 안 함 | "개인 식별 정보는 조회할 수 없습니다." 칩: 대표자 수만 집계 / 업종별 점포 목록 |

이 표가 그대로 `run.py`의 시나리오와 최종 상태 단정이 됩니다(예: 턴 5 뒤 `task_stack`에 PII 컬럼을 읽은 SQL이 없다, 턴 4 뒤 `user_profile`에 확정 행 1개).

---

## 7. 공공 보안 기준 서버 · 모듈 배치

이 저장소는 이미 공공 환경으로 옮기기 쉬운 형태입니다. `llm.py`가 OpenAI 호환 `ChatOpenAI`라 `base_url`만 내부 추론 서버로 바꾸면 되고, 트레이싱(Langfuse v3)도 자체 호스팅입니다.

| 계층 | 지금 | 공공 배치 |
|---|---|---|
| LLM | OpenRouter (Sonnet/Haiku) | 내부망 vLLM 등 OpenAI 호환 서버. 역할별 모델: 의도 추출·칩(소형), 판단·보고서(대형), VLM(첨부), 코드 특화(SQL) |
| 그래프 · 에이전트 | 로컬 프로세스 | 앱 서버(CPU) — FastAPI + LangGraph, 메모리 추출 워커 별도 프로세스 |
| 도구 | MCP stdio 목 서버 | 업무 시스템별 MCP 서버(내부망), DB는 읽기 전용 계정 |
| 상태 | `InMemorySaver`, JSON 파일 | Postgres(체크포인트 · 프로필 · 스키마 인덱스 pgvector) + Redis(화면 세션) |
| 관측 | Langfuse 로컬 | 같은 스택을 내부망에, 트레이스 내 PII 마스킹 후 저장 |
| 암호 · 망 | — | 저장 PII는 KCMVP 검증 암호모듈, 구간 TLS, 인터넷망 연계는 망연계 솔루션 경유, CSAP 인증 클라우드 또는 온프레미스 |

GPU 산정은 참고안("H100 1대로 70B")보다 보수적으로 잡아야 합니다. 70B급을 FP16으로 올리면 가중치만 약 140GB라 80GB 1장에 들어가지 않습니다. 양자화(INT4/FP8) 또는 2장 이상, 그리고 VLM · SQL 모델을 따로 띄운다면 그만큼 추가가 필요합니다. 모델 확정 후 동시 사용자 수와 목표 지연(첫 글자 시간)으로 다시 계산합니다.

---

## 8. 구현 순서

1. **상태 · 규칙 일반화** — 카탈로그 `produces/consumes`, `refine`·`remember` op, 가이드라인 레지스트리. 전부 `state.py` 순수 함수라 테스트 먼저.
2. **Text-to-SQL** — 목 DB 2개(MCP), 스키마 인덱스, 검증기(sqlglot), `sql_agent`. 모호성 칩 · 비용 확인 칩 · 실패 복구 칩 시나리오.
3. **정책 가드** — `input_guard`/`output_guard`, PII 컬럼 차단, `policy_flags`. 우회 시나리오 E2E.
4. **멀티모달** — `attachment_reader`, 오브젝트 스토리지 참조, 첨부 사실 → 슬롯 연결.
5. **장기 메모리** — 프로필 테이블, 비동기 추출, 확인 칩, 정정·삭제 API.
6. **공공 배치** — `PostgresSaver`, 내부 추론 서버 전환, 모델별 지연·정확도 재측정(`inspect_traces`로 턴별 비교).

각 단계는 독립 E2E 시나리오(6장 표의 한 줄)를 하나씩 추가하며 끝냅니다.

### 구현 현황 (2026-11-18)

- 1~5단계 구현 완료. 6단계는 LLM 엔드포인트 전환(`LLM_BASE_URL`)과 체크포인터 주입 지점까지.
- 테스트 153개 (LLM 없음). 실제 그래프 + MCP 서버로 첨부 → 출처 칩 → SQL → 보고서 → refine 흐름을 가짜 LLM으로 검증 (`tests/test_graph_flow.py`).
- 내부 Solar 실측 14/14 통과 (2026-11-19). VLM이 없어 이미지는 Document Parse OCR → Solar 구조화 경로로 판독하고, 새 첨부에는 규칙으로 해석 작업을 붙인다.
- 작은 모델 대응으로 설계에 추가된 것:
  - 2단계 의도 추출: 의도 분리 → 의도별 작은 스키마(json_schema)로 슬롯을 병렬로 채움
  - 근거 단어가 없는 슬롯(출처·범위·기간) 제거
  - 실행 직전 필수 슬롯 재확인
  - 구조 기반 refine 판단: 모델이 `existing_task_id`를 못 채워도 동작
- 5.7의 `set_preference` 위험도는 low로 두었습니다. 동의 칩이 확인 역할을 하기 때문입니다. 답하지 않은 동의 칩은 2턴 뒤 자동으로 취소됩니다.

---

## 9. 열린 질문

- 제안요청서의 대상 DB 목록과 스키마 규모(테이블 수, 테이블 정의서 품질) — 스키마 인덱스 설계와 ① 단계 정확도에 직결
- 백엔드 모델 제약(국산 모델 지정 여부, GPU 수량) — 역할별 모델 분리 여부 결정
- 장기 메모리 보존 기간 · 동의 절차의 기관 기준
- 보고서 양식(가이드라인) 원본의 형태 — 문서 템플릿인지, 규칙 목록인지
