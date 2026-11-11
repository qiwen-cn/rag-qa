import os

from fastapi import HTTPException
import requests

# DeepSeek API的基础URL和模型名称
base_url = "https://api.deepseek.com/chat/completions"
model = "deepseek-chat"

# 读取环境变量中的LLM_API_KEY
api_key = os.environ.get("LLM_API_KEY")
if not api_key:
    raise RuntimeError("请设置环境变量LLM_API_KEY")


def get_chunks():
    """读取本地文档，并将其切分成一段段适合大模型阅读的文本块 (Chunk)"""
    with open("D:/workspace/code/rag-qa/doc.txt", "r", encoding="UTF-8") as f:
        document = f.read()

        # 按双换行符（空行）把整篇文章切成一个个自然段
        paragraphs = document.split("\n\n")
        chunks = []
        buffer = ""  # 收集箱：用于把短小的段落拼凑到一起

        for p in paragraphs:
            p = p.strip()
            if not p:  # 跳过空行
                continue

            # 将当前段落拼接到收集箱（如果收集箱非空，中间加个换行符）
            buffer = (buffer + "\n" + p) if buffer else p

            # 过滤：如果是Markdown标题（#开头），或者字数还没攒够40字，就跳过不打包
            if p.startswith("#") or len(buffer) <= 40:
                continue

            # # 攒够了40个字，就打包作为一个Chunk
            chunks.append(buffer)
            buffer = ""

        # 循环结束后，如果收集箱还有剩余内容，也打包进去
        if buffer:
            chunks.append(buffer)
    return chunks


def retrieve(chunks, question, top_k=3):
    """简易检索器：根据问题中的字词在文本块中出现的次数来打分，找出最相关的 top_k 个块"""
    scored = []

    for c in chunks:
        # 核心打分逻辑：提取问题中的去重字符（只保留字母和数字），计算它们在这个文本块里出现了多少次
        s = sum(c.count(q) for q in set(question) if q.isalnum())
        if s > 0:  # 只有得分大于0的文本块才保留
            scored.append((s, c))

    # 按得分从高到低排序
    scored.sort(key=lambda x: x[0], reverse=True)

    # 返回得分最高的前几个文本块
    return scored[:top_k]


def ask(question):
    """核心问答函数：检索相关文本 -> 拼接 Prompt -> 调用大模型 -> 返回答案"""
    # 检索：获取与问题相关的3个文本块
    top = retrieve(get_chunks(), question, 3)

    # 拼接上下文：给检索出的文本块加上[1]、[2]这样的编号，方便溯源
    context = "\n\n".join([f"[{i+1}]：{chunk}" for i, (score, chunk) in enumerate(top)])

    # 设置请求头
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    # 构造请求体
    body = {
        "model": model,
        "messages": [
            # 核心Prompt设定：要求大模型只能根据资料回答，不能产生幻觉
            {
                "role": "system",
                "content": "你是文档问答助手。只能根据提供的资料回答。资料里没有的，就回答'资料中没有提到'，不要编造。回答用纯文本，不要使用 Markdown 格式。",
            },
            # 把检索到的资料和用户的问题一起发给大模型
            {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
        ],
    }

    # 发送请求给 DeepSeek
    response = requests.post(
        base_url,
        json=body,
        headers=headers,
        timeout=90,
    )

    # 异常处理：如果上游报错，抛出标准HTTP异常
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"上游服务调用失败，状态码 {response.status_code}",
        )

    # 解析结果，提取大模型的回复文本
    reply = response.json()["choices"][0]["message"]["content"]

    # 返回最终答案，以及它所依据的上下文资料
    # return {"reply": reply, "context": context}
    return {"answer": reply, "sources": [c for s, c in top]}  # 由前端决定怎么展示


if __name__ == "__main__":
    print(ask("这个项目每次检索返回几段？")["answer"])
