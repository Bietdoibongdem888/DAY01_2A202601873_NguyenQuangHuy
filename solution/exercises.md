# K3 — Ngày 1: Bài Tập & Phản Ánh

## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature

Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Ở temperature thấp (0.0) mô hình trả lời rất determinist, giống nhau và ít sáng tạo. Ở mức trung bình (0.5–1.0) câu trả lời đa dạng hơn, có thêm ví dụ và từ ngữ khác nhau. Ở 1.5 kết quả ngẫu nhiên nhiều, có thể mất tính chính xác hoặc sinh nội dung bất thường.

### Câu 1.2 — Chọn temperature cho sản phẩm

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Khoảng 0.0–0.3: ưu tiên tính nhất quán, chính xác và dễ kiểm soát; giảm rủi ro sinh thông tin sai. Nếu cần trả lời thân thiện hơn có thể tăng nhẹ tới 0.5.

### Câu 1.3 — Đánh đổi chi phí

Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Workload: 10.000 * 3 * 350 = 10_500_000 token output. Với giá lần lượt 0.010 USD/1K (gpt-4o) và 0.0006 USD/1K (mini):
> - gpt-4o: 10_500_000/1000 * 0.010 = 105 USD
> - gpt-4o-mini: 10_500_000/1000 * 0.0006 = 6.3 USD
> => gpt-4o ~16.7x đắt hơn mini.
> Khi cần hiểu sâu, reasoning phức tạp, logic hoặc chất lượng ngôn ngữ tinh tế (ví dụ tư vấn pháp lý, lập luận chuyên sâu) thì gpt-4o xứng đáng. Cho các tác vụ trả lời FAQ, tóm tắt ngắn, hoặc scale lớn mà yêu cầu chất lượng vừa phải thì dùng mini.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona

Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau.

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Persona thay đổi giọng điệu, độ dài và từ vựng: system prompt "giáo viên tiểu học" cho câu trả lời ngắn gọn, dùng từ đơn giản và ví dụ trực quan; prompt "chuyên gia tài chính" thì dùng thuật ngữ chuyên môn, giải thích sâu và chi tiết. System prompt định hướng style, mức chi tiết và lựa chọn ví dụ của mô hình.

### Câu 2.2 — tiktoken vs đếm từ

Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Thường tiktoken cho số token cao hơn ước lượng từ/0.75 (ví dụ chênh 10–30%). Nguyên nhân: tokenizer phân mảnh từ và mã hóa subword; tiếng Việt có nhiều từ ngắn và dấu câu/kiểu từ khiến phân mảnh không giống tiếng Anh, cộng thêm biểu diễn UTF-8 khiến một số ký tự có byte/encoding làm tăng token so với đơn thuần đếm từ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming

**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng khi câu trả lời dài hoặc khi người dùng cần phản hồi sớm (chat realtime, assistant trả lời từng phần) để giảm độ trễ cảm nhận. Non-streaming phù hợp cho các tác vụ ngắn, muốn kết quả nguyên khối (ví dụ trả về JSON hoàn chỉnh) hoặc khi cần xử lý/kiểm tra nội dung trước khi hiển thị.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?

**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giảm tần suất retry theo thời gian, giúp hệ thống ổn định dần và tránh quá tải tiếp tục. Nếu mọi client retry với delay cố định giống nhau sẽ tạo ra "thundering herd" — đồng loạt gửi lại gây tăng tải và làm mọi thứ tệ hơn; backoff phá vỡ sự đồng bộ này.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona

**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt:**
> Persona: "Bạn là trợ giảng thân thiện, trả lời ngắn gọn, rõ ràng bằng tiếng Việt, ưu tiên cung cấp ví dụ thực tế và bước thực hiện".
> - "trả lời ngắn gọn" để giữ độ dài phù hợp với UI chat và giảm chi phí token. 
> - "ưu tiên ví dụ" giúp người học hiểu nhanh qua minh họa.

### Câu 4.2 — Hạn chế & cải thiện

**Trợ lý của bạn hiện có hạn chế lớn nhất là gì? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế: history ngắn (3 lượt) nên mất ngữ cảnh dài hạn. Cải thiện: thêm bộ nhớ ngắn hạn/ dài hạn bằng cách lưu embedding của các lượt hội thoại vào vector DB (ví dụ FAISS) và khi bắt đầu phiên hoặc trước mỗi lượt, truy vấn các đoạn có relevance cao để bổ sung vào system prompt. Triển khai: tính embedding cho mỗi reply, lưu vào DB, truy vấn top-k theo cosine, đưa kết quả vào messages làm context tham chiếu.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
