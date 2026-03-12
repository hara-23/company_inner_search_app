"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import csv
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_core.documents import Document as LCDocument
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv(dotenv_path=os.path.join(ct.BASE_DIR, ".env"))


############################################################
# 関数定義
############################################################

def get_source_icon(source):
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    # 参照元がWebページの場合とファイルの場合で、取得するアイコンの種類を変える
    if source.startswith("http"):
        icon = ct.LINK_SOURCE_ICON
    else:
        icon = ct.DOC_SOURCE_ICON
    
    return icon


def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def build_source_message(source, page_number=None):
    """
    参照元の表示テキストを作成

    Args:
        source: 参照元のありか
        page_number: 参照ページ番号（0始まり想定）

    Returns:
        画面表示用の参照元テキスト
    """
    if source.lower().endswith(".pdf") and page_number is not None:
        return f"{source}（ページNo.{page_number + 1}）"

    return source


def get_llm_response(chat_message):
    """
    LLMからの回答取得

    Args:
        chat_message: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # 「人事部の従業員一覧」系は、CSVを直接参照するフォールバックで安定化
    fallback_response = get_hr_employee_list_response(chat_message)
    if fallback_response:
        st.session_state.chat_history.extend([HumanMessage(content=chat_message), fallback_response["answer"]])
        return fallback_response

    # LLMのオブジェクトを用意
    llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのプロンプトテンプレートを作成
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # モードによってLLMから回答を取得する用のプロンプトを変更
    if st.session_state.mode == ct.ANSWER_MODE_1:
        # モードが「社内文書検索」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        # モードが「社内問い合わせ」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    # LLMから回答を取得する用のプロンプトテンプレートを作成
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのRetrieverを作成
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, question_generator_prompt
    )

    # LLMから回答を取得する用のChainを作成
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    # 「RAG x 会話履歴の記憶機能」を実現するためのChainを作成
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # LLMへのリクエストとレスポンス取得
    llm_response = chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
    # LLMレスポンスを会話履歴に追加
    st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response["answer"]])

    return llm_response


def get_hr_employee_list_response(chat_message):
    """
    人事部の従業員一覧質問に対するCSV直読フォールバック

    Args:
        chat_message: ユーザー入力値

    Returns:
        フォールバック回答（該当しない場合はNone）
    """
    if st.session_state.mode != ct.ANSWER_MODE_2:
        return None

    normalized = chat_message.replace(" ", "")
    if "人事" not in normalized or ("従業員" not in normalized and "社員" not in normalized):
        return None

    csv_path = os.path.join(ct.RAG_TOP_FOLDER_PATH, "社員について", "社員名簿.csv")
    if not os.path.exists(csv_path):
        return None

    # UTF-8 / CP932 の両方に対応
    rows = None
    for enc in ["utf-8", "cp932"]:
        try:
            with open(csv_path, "r", encoding=enc, newline="") as f:
                rows = list(csv.DictReader(f))
            break
        except Exception:
            continue

    if rows is None:
        return None

    hr_rows = [row for row in rows if "人事" in (row.get("部署", "") + row.get("所属", ""))]
    if not hr_rows:
        return None

    display_columns = ["社員ID", "氏名（フルネーム）", "部署", "役職", "メールアドレス"]
    markdown_lines = [
        "### 人事部に所属している従業員情報",
        "",
        "| " + " | ".join(display_columns) + " |",
        "|" + "|".join(["---"] * len(display_columns)) + "|",
    ]

    for row in hr_rows:
        markdown_lines.append("| " + " | ".join([str(row.get(col, "")).strip() for col in display_columns]) + " |")

    markdown_lines.append("")
    markdown_lines.append(f"現在、人事部に所属している従業員は {len(hr_rows)} 名です。")

    return {
        "answer": "\n".join(markdown_lines),
        "context": [LCDocument(page_content="社員名簿CSV（人事部抽出）", metadata={"source": csv_path})],
    }