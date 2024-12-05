# Mô hình ZeroToolBERT

> Bài toán function calling là một trong những bài toán quan trọng của các mô hình ngôn ngữ lớn (LLM). Tuy nhiên, các phương pháp hiện tại thường kém hiệu quả và yêu cầu tài nguyên tính toán lớn do phải chèn tất cả các công cụ vào prompt của mô hình. Để giải quyết vấn đề này, chúng tôi đề xuất ZeroToolBERT, một mô hình có khả năng lựa chọn các công cụ cần thiết để đáp ứng yêu cầu của người dùng thay vì sử dụng toàn bộ. Phương pháp của chúng tôi sử dụng BERT để tạo biểu diễn văn bản và được thử nghiệm trên hai kiến trúc phổ biến: MLP và Attention. Kết quả thực nghiệm cho thấy mô hình đạt độ chính xác 95% và precision 96%.

## Cấu trúc mô hình

<p align="center">
 <img src="https://github.com/user-attachments/assets/40b3a9e7-9dfe-4857-9900-259c5c441a5a" />
</p>

## Kết quả

<p align="center">
 <img src="https://github.com/user-attachments/assets/1067df3c-397d-4ff6-8a1b-8e476d4560c3" />
</p>



