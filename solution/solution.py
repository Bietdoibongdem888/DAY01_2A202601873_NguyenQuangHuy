"""
K3 — Ngày 1: Khám Phá LLM API (9h00–13h00)
AICB-P1: AI Practical Competency Program, Phase 1

Hướng dẫn:
    1. Làm theo LAB_GUIDE.md — mỗi block có các bước chi tiết và checkpoint.
    2. Điền vào tất cả các chỗ đánh dấu TODO.
    3. KHÔNG đổi chữ ký hàm (tên hàm, tham số).
    4. Import OpenAI BÊN TRONG hàm (xem gợi ý) — nếu import ở đầu file,
       các bài test mock sẽ không hoạt động.
    5. Kiểm tra tiến độ:  pytest tests/test_part1.py -v  (từng phần)
       Chấm điểm tổng:    python grade.py
"""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

# Nạp OPENAI_API_KEY từ file .env (copy .env.example thành .env và dán key vào)
load_dotenv()

# ---------------------------------------------------------------------------
# Bảng giá ước tính (USD / 1K token) — cập nhật nếu giá thay đổi
# ---------------------------------------------------------------------------
PRICING_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

# Tên model có thể đổi qua .env — ví dụ khi dùng NVIDIA NIM miễn phí
# (xem LAB_GUIDE.md, Phụ lục B). Không đặt gì trong .env thì mặc định OpenAI.
OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")
OPENAI_MINI_MODEL = os.getenv("LAB_MINI_MODEL", "gpt-4o-mini")


# ===========================================================================
# PART 1 — API CƠ BẢN (Block 1: 10h00–10h40)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 1.1 — Gọi GPT-4o
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API chat completion cơ bản và trả về (text, latency_seconds).
    """
    from openai import OpenAI  # import BÊN TRONG hàm

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start

    # Thử lấy text theo cấu trúc mock trong tests: resp.choices[0].message.content
    try:
        text = resp.choices[0].message.content
    except Exception:
        try:
            text = resp.choices[0].text
        except Exception:
            text = str(resp)

    return text, float(latency)


# ---------------------------------------------------------------------------
# Task 1.2 — Gọi GPT-4o-mini
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với model gpt-4o-mini — nhanh hơn và rẻ hơn.

    Returns:
        Tuple (response_text: str, latency_seconds: float).
    """
    # Tái sử dụng call_openai với model mini
    return call_openai(
        prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Task 1.3 — So sánh GPT-4o vs GPT-4o-mini
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Gọi cả hai model với cùng một prompt và trả về dict so sánh.
    """
    gpt4o_text, gpt4o_latency = call_openai(prompt)
    mini_text, mini_latency = call_openai_mini(prompt)

    # ước lượng thô theo gợi ý (tính theo số từ)
    try:
        word_count = len(gpt4o_text.split())
    except Exception:
        word_count = 0

    pricing = PRICING_PER_1K_TOKENS.get("gpt-4o", PRICING_PER_1K_TOKENS["gpt-4o"])
    gpt4o_cost_estimate = (word_count / 0.75) / 1000 * pricing["output"]

    return {
        "gpt4o_response": gpt4o_text,
        "mini_response": mini_text,
        "gpt4o_latency": float(gpt4o_latency),
        "mini_latency": float(mini_latency),
        "gpt4o_cost_estimate": float(gpt4o_cost_estimate),
    }


# ===========================================================================
# PART 2 — SYSTEM PROMPT & TOKEN (Block 2: 10h40–11h20)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 2.1 — Chat với system prompt (persona)
# ---------------------------------------------------------------------------
def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với MESSAGES gồm system + user và trả về (text, latency).
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.time() - start

    try:
        text = resp.choices[0].message.content
    except Exception:
        try:
            text = resp.choices[0].text
        except Exception:
            text = str(resp)

    return text, float(latency)


# ---------------------------------------------------------------------------
# Task 2.2 — Đếm token bằng tiktoken
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    """
    Đếm số token của một đoạn text bằng thư viện tiktoken (hoặc fallback).
    """
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # Fallback ước lượng: trung bình ~4 ký tự/ token
        try:
            return max(1, len(text) // 4)
        except Exception:
            return 1


# ---------------------------------------------------------------------------
# Task 2.3 — Ước tính chi phí chính xác
# ---------------------------------------------------------------------------
def estimate_cost(prompt: str, response: str, model: str = OPENAI_MODEL) -> dict:
    """
    Tính chi phí một lượt gọi API dựa trên số token thực tế.
    """
    input_tokens = int(count_tokens(prompt, model=model))
    output_tokens = int(count_tokens(response, model=model))

    pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS["gpt-4o"]) 

    input_cost = input_tokens / 1000.0 * pricing.get("input", PRICING_PER_1K_TOKENS["gpt-4o"]["input"]) 
    output_cost = output_tokens / 1000.0 * pricing.get("output", PRICING_PER_1K_TOKENS["gpt-4o"]["output"]) 

    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": float(input_cost),
        "output_cost": float(output_cost),
        "total_cost": float(total_cost),
    }


# ===========================================================================
# PART 3 — STREAMING & ĐỘ BỀN (Block 3: 11h30–12h10)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 3.1 — Chatbot streaming có lịch sử hội thoại
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Chatbot dòng lệnh tương tác dùng streaming.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    history = []  # list of messages dicts

    while True:
        user_msg = input()
        if user_msg.strip().lower() in ("quit", "exit"):
            break

        # build messages: just use history + user
        messages = history + [{"role": "user", "content": user_msg}]

        stream = client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, stream=True
        )

        # stream chunks and print
        reply_parts = []
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content or ""
            except Exception:
                try:
                    delta = chunk.choices[0].message.content or ""
                except Exception:
                    delta = ""
            print(delta, end="", flush=True)
            reply_parts.append(delta)
        print()  # newline after reply

        reply = "".join([p for p in reply_parts if p])

        # update history: user then assistant
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})

        # keep only last 6 messages (3 turns)
        history = history[-6:]


# ---------------------------------------------------------------------------
# Task 3.2 — Retry với exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Gọi fn() với retry và exponential backoff.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            if attempt >= max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
            attempt += 1


# ===========================================================================
# PART 4 — MINI-PROJECT: TRỢ LÝ CLI HOÀN CHỈNH (Block 4: 12h10–12h50)
# ===========================================================================
def run_assistant(
    persona: str,
    get_input: Callable[[], str] = None,
    max_turns: int = None,
) -> dict:
    """
    Trợ lý CLI hoàn chỉnh — ghép mọi thứ bạn đã xây trong Part 1–3.

    Hành vi:
        1. Dùng `persona` làm system prompt cho TOÀN BỘ phiên chat.
        2. Mỗi lượt: đọc tin nhắn qua get_input(); nếu là 'quit'/'exit'
           (không phân biệt hoa thường) → kết thúc phiên.
        3. Gọi API với stream=True, messages = system + history + tin nhắn mới.
           Bọc lời gọi API trong retry_with_backoff để chịu lỗi tạm thời.
        4. In từng chunk khi stream về, ghép lại thành reply hoàn chỉnh.
        5. Cập nhật history (user + assistant), giữ tối đa 3 lượt cuối
           (6 message): history = history[-6:]
        6. Cộng dồn thống kê bằng count_tokens và estimate_cost.
        7. Dừng khi đạt max_turns (nếu được đặt).

    Args:
        persona:   Mô tả vai trò, dùng làm system prompt.
        get_input: Hàm đọc input (mặc định: input). Tham số này giúp
                   test tự động không cần bàn phím thật.
        max_turns: Số lượt tối đa (None = không giới hạn).

    Returns:
        Dict thống kê phiên chat:
            - "num_turns":    int   (số lượt hỏi–đáp đã thực hiện)
            - "total_tokens": int   (tổng token user + assistant)
            - "total_cost":   float (tổng USD ước tính)
            - "history":      list  (history còn lại sau khi cắt, ≤ 6 message)

    Gợi ý khung sườn:
        if get_input is None:
            get_input = input
        history, num_turns, total_tokens, total_cost = [], 0, 0, 0.0
        while True:
            if max_turns is not None and num_turns >= max_turns:
                break
            user_msg = get_input()
            if user_msg.strip().lower() in ("quit", "exit"):
                break
            messages = [{"role": "system", "content": persona}] + history \\
                       + [{"role": "user", "content": user_msg}]
            # stream = retry_with_backoff(lambda: client.chat...create(
            #              model=..., messages=messages, stream=True))
            # reply = ghép các chunk...
            ...
        return {"num_turns": num_turns, "total_tokens": total_tokens,
                "total_cost": total_cost, "history": history}
    """
    # TODO: triển khai theo khung sườn trong docstring

    from openai import OpenAI

    if get_input is None:
        get_input = input

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    history = []
    num_turns = 0
    total_tokens = 0
    total_cost = 0.0

    while True:
        if max_turns is not None and num_turns >= max_turns:
            break
        try:
            user_msg = get_input()
        except Exception:
            break
        if user_msg.strip().lower() in ("quit", "exit"):
            break

        messages = [{"role": "system", "content": persona}] + history + [{"role": "user", "content": user_msg}]

        def call_api():
            return client.chat.completions.create(model=OPENAI_MODEL, messages=messages, stream=True)

        stream = retry_with_backoff(call_api)

        reply_parts = []
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content or ""
            except Exception:
                try:
                    delta = chunk.choices[0].message.content or ""
                except Exception:
                    delta = ""
            print(delta, end="", flush=True)
            reply_parts.append(delta)
        print()

        reply = "".join([p for p in reply_parts if p])

        num_turns += 1
        total_tokens += int(count_tokens(user_msg))
        total_tokens += int(count_tokens(reply))
        cost_info = estimate_cost(user_msg, reply, model=OPENAI_MODEL)
        total_cost += float(cost_info.get("total_cost", 0.0))

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})
        history = history[-6:]

    return {"num_turns": num_turns, "total_tokens": total_tokens,
            "total_cost": total_cost, "history": history}


# ===========================================================================
# BONUS (không bắt buộc — cho bạn nào xong sớm)
# ===========================================================================
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Chạy compare_models cho từng prompt trong list.

    Returns:
        List các dict — mỗi dict là kết quả compare_models kèm thêm
        key "prompt" chứa prompt gốc.
    """
    results = []
    for p in prompts:
        try:
            res = compare_models(p)
        except Exception:
            res = {}
        res["prompt"] = p
        results.append(res)
    return results


def format_comparison_table(results: list[dict]) -> str:
    """
    Định dạng kết quả batch_compare thành bảng text dễ đọc.

    Cột: Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency
    Gợi ý: cắt text dài còn 40 ký tự cho dễ nhìn.
    """
    def _short(s: str, n: int = 40) -> str:
        if s is None:
            return ""
        s = str(s).replace("\n", " ")
        return (s[: n - 3] + "...") if len(s) > n else s

    lines = []
    header = ["Prompt", "GPT-4o Response", "Mini Response", "GPT-4o Latency", "Mini Latency"]
    lines.append(" | ".join(header))
    lines.append("-" * 80)

    for r in results:
        row = [
            _short(r.get("prompt", ""), 30),
            _short(r.get("gpt4o_response", ""), 40),
            _short(r.get("mini_response", ""), 40),
            f"{r.get('gpt4o_latency', '')}",
            f"{r.get('mini_latency', '')}",
        ]
        lines.append(" | ".join(row))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point — demo chạy thật (cần OPENAI_API_KEY)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== So sánh model ===")
    result = compare_models(
        "Giải thích khác biệt giữa temperature và top_p trong một câu."
    )
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Trợ lý CLI (gõ 'quit' để thoát) ===")
    stats = run_assistant(
        persona="Bạn là trợ giảng thân thiện của khóa AI, "
                "trả lời ngắn gọn bằng tiếng Việt.",
    )
    print("\n--- Thống kê phiên chat ---")
    for key, value in stats.items():
        if key != "history":
            print(f"{key}: {value}")
