const renderer = {
    // 링크는 새창에서 열리도록 함
    link(href, title, text){
        // marked.js 기본 링크 렌더러 호출
        const link = marked.Renderer.prototype.link.call(this, href, title, text);
        return link.replace("<a", "<a target='_blank' rel='noreferrer'")
    }
};

marked.use({
    renderer,
});

// 고유한 세션아이디 생성
const generateSessionId = () => {
    const timestamp = Date.now();
    const randomString = Math.random().toString(36).substring(2,9);
    return `session_${timestamp}_${randomString}`;
}

// 채팅 어플리케이션 관리 모듈
const ChatApp = {
    elements: {
        chatForm: null,
        chatInput: null,
        chatBox: null,
        chatImg: null,
    },
    sessionId: null,
    init(){
        // dom 요소들을 찾아 저장
        this.elements.chatForm = document.getElementById("chat-form");
        this.elements.chatInput = document.getElementById("chat-input");
        this.elements.chatBox = document.getElementById("chat-box");

        // 세션 id 생성, 로그에 기록
        this.sessionId = generateSessionId();
        console.log("새로운 세션 ID:", this.sessionId);

        // 이벤트 리스너 등록
        this.elements.chatForm.addEventListener(
            "submit",
            this.handleFormSubmit.bind(this)
        );
    },
    async handleFormSubmit(e){
        // 채팅 폼 제출 이벤트 처리
        e.preventDefault();
        const message = this.elements.chatInput.value.trim();
        if(!message) return;

        const fd = new FormData();
        fd.append("session_id", this.sessionId);
        fd.append("message", message);

        // 사용자 메시지를 화면에 추가
        this.appendMessage("user", message);
        this.elements.chatInput.value = "";

        // 봇의 응답을 스트리밍
        const botMessageElement = this.createMessageElement("bot");
        await this.streamBotResponse(fd, botMessageElement);
    },
    async streamBotResponse(fd, botMessageElement){
        // 서버의 스트리밍 응답 처리
        try {
            const response = await fetch("/chat", {
                method: "POST",
                //headers: {"Content-Type": "application/x-www-form-urlencoded"},
                body: fd
            });

            if (!response.ok) {
                throw new Error(`HTTP Error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let content = "";

            // stream 읽어 화면 표시
            while (true) {
                const {value, done} = await reader.read();
                if (done) break;

                content += decoder.decode(value, {stream: true});
                botMessageElement.innerHTML = marked.parse(content);
                this.scrollToBottom();
            }
        }catch(error) {
            console.error("스트리밍 중 오류 발생:", error);
            botMessageElement.innerHTML = "죄송합니다. 메시지를 처리하는 중 오류가 발생했습니다.";
        }
    },
    createMessageElement(sender){
        // 새로운 메시지 요소를 생성하고 dom에 추가
        const messageElement= document.createElement("div");
        messageElement.classList.add("message", `${sender}-message`);
        this.elements.chatBox.appendChild(messageElement);
        this.scrollToBottom();
        return messageElement;
    },
    appendMessage(sender, text){
        // 메시지를 화면에 추가합니다.
        const messageElement = this.createMessageElement(sender);
        messageElement.innerHTML = marked.parse(text);
    },
    scrollToBottom(){
        // 채팅 박스를 맨 아래로 스크롤
        this.elements.chatBox.scrollTop = this.elements.chatBox.scrollHeight;
    }
};

// dom 로드 후 애플리케이션 초기화
document.addEventListener("DOMContentLoaded", () => {
   ChatApp.init();
});
