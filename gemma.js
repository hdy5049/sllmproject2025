document.addEventListener("DOMContentLoaded", () => {
  const submitButton = document.getElementById("submit");
  const inputField = document.getElementById("input");
  const responseDiv = document.getElementById("response");

  submitButton.addEventListener("click", async () => {
    const input = inputField.value.trim();
    if (!input) {
      responseDiv.innerText = "질문을 입력해주세요.";
      return;
    }

    responseDiv.innerText = "응답 생성 중...";

    const startTime = performance.now(); // 시작 시간 기록

    try {
      const res = await fetch("http://127.0.0.1:8000/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: input })
      });

      const endTime = performance.now(); // 끝난 시간 기록
      const duration = ((endTime - startTime) / 1000).toFixed(2); // 초로 계산

      const data = await res.json();
      responseDiv.innerText = `${data.response}\n\n⏱ ${duration}초 소요됨`;
    } catch (error) {
      responseDiv.innerText = "서버 요청 중 오류가 발생했습니다";
      console.error(error);
    }
  });
});
