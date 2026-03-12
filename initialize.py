"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import csv
import logging
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st
from docx import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document as LCDocument
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import constants as ct


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv(dotenv_path=os.path.join(ct.BASE_DIR, ".env"))


def get_secret_value(key):
    """
    Streamlit secrets から値を安全に取得
    secrets.toml が未配置でも例外を出さず None を返す
    """
    # ローカル実行時に secrets.toml が無い場合、st.secrets 参照自体を行わない
    secret_file_candidates = [
        Path.home() / ".streamlit" / "secrets.toml",
        Path(ct.BASE_DIR) / ".streamlit" / "secrets.toml",
    ]
    is_cloud_runtime = bool(os.getenv("STREAMLIT_SHARING_MODE"))

    if not is_cloud_runtime and not any(path.exists() for path in secret_file_candidates):
        return None

    try:
        return st.secrets.get(key)
    except Exception:
        return None

# Streamlit Cloudでは .env を使わず App settings (secrets) を利用する
if not os.getenv("OPENAI_API_KEY"):
    secret_openai_api_key = get_secret_value("OPENAI_API_KEY")
    if secret_openai_api_key:
        os.environ["OPENAI_API_KEY"] = secret_openai_api_key

if not os.getenv("USER_AGENT"):
    secret_user_agent = get_secret_value("USER_AGENT")
    if secret_user_agent:
        os.environ["USER_AGENT"] = secret_user_agent


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGのRetrieverを作成
    try:
        initialize_retriever()
        st.session_state.retriever_init_error = None
    except Exception as e:
        logger = logging.getLogger(ct.LOGGER_NAME)
        logger.error(f"Retriever初期化に失敗したため、起動は継続します。\n{e}")
        st.session_state.retriever = None
        st.session_state.retriever_init_error = str(e)


def initialize_logger():
    """
    ログ出力の設定
    """
    # 指定のログフォルダが存在すれば読み込み、存在しなければ新規作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    # 引数に指定した名前のロガー（ログを記録するオブジェクト）を取得
    # 再度別の箇所で呼び出した場合、すでに同じ名前のロガーが存在していれば読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにロガーにハンドラー（ログの出力先を制御するもの）が設定されている場合、同じログ出力が複数回行われないよう処理を中断する
    if logger.hasHandlers():
        return

    # 1日単位でログファイルの中身をリセットし、切り替える設定
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    # 出力するログメッセージのフォーマット定義
    # - 「levelname」: ログの重要度（INFO, WARNING, ERRORなど）
    # - 「asctime」: ログのタイムスタンプ（いつ記録されたか）
    # - 「lineno」: ログが出力されたファイルの行番号
    # - 「funcName」: ログが出力された関数名
    # - 「session_id」: セッションID（誰のアプリ操作か分かるように）
    # - 「message」: ログメッセージ
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    # 定義したフォーマッターの適用
    log_handler.setFormatter(formatter)

    # ログレベルを「INFO」に設定
    logger.setLevel(logging.INFO)

    # 作成したハンドラー（ログ出力先を制御するオブジェクト）を、
    # ロガー（ログメッセージを実際に生成するオブジェクト）に追加してログ出力の最終設定
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    """
    画面読み込み時にRAGのRetriever（ベクターストアから検索するオブジェクト）を作成
    """
    # ロガーを読み込むことで、後続の処理中に発生したエラーなどがログファイルに記録される
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにRetrieverが作成済みの場合、後続の処理を中断
    if "retriever" in st.session_state:
        return
    
    # RAGの参照先となるデータソースの読み込み
    docs_all = load_data_sources()

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    # 埋め込みモデルの用意
    embeddings = OpenAIEmbeddings()
    
    # チャンク分割用のオブジェクトを作成
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.RAG_CHUNK_SIZE,
        chunk_overlap=ct.RAG_CHUNK_OVERLAP,
        separator="\n"
    )

    # チャンク分割を実施
    # CSVは一覧性を維持するため、1ファイル=1ドキュメントのまま保持する
    splitted_docs = []
    for doc in docs_all:
        if doc.metadata.get("csv_merged"):
            splitted_docs.append(doc)
            continue
        splitted_docs.extend(text_splitter.split_documents([doc]))

    # ベクターストアの作成
    db = Chroma.from_documents(splitted_docs, embedding=embeddings)

    # ベクターストアを検索するRetrieverの作成
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": ct.RETRIEVER_TOP_K})


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        st.session_state.chat_history = []


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース
    """
    # データソースを格納する用のリスト
    docs_all = []
    # ファイル読み込みの実行（渡した各リストにデータが格納される）
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    web_docs_all = []
    # ファイルとは別に、指定のWebページ内のデータも読み込み
    # 読み込み対象のWebページ一覧に対して処理
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        # 指定のWebページを読み込み
        try:
            loader = WebBaseLoader(web_url)
            web_docs = loader.load()
            # for文の外のリストに読み込んだデータソースを追加
            web_docs_all.extend(web_docs)
        except Exception:
            # 外部サイトの一時的な接続不良時は、ローカルデータのみで継続する
            continue
    # 通常読み込みのデータソースにWebページのデータを追加
    docs_all.extend(web_docs_all)

    return docs_all


def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    # パスがフォルダかどうかを確認
    if os.path.isdir(path):
        # フォルダの場合、フォルダ内のファイル/フォルダ名の一覧を取得
        files = os.listdir(path)
        # 各ファイル/フォルダに対して処理
        for file in files:
            # ファイル/フォルダ名だけでなく、フルパスを取得
            full_path = os.path.join(path, file)
            # フルパスを渡し、再帰的にファイル読み込みの関数を実行
            recursive_file_check(full_path, docs_all)
    else:
        # パスがファイルの場合、ファイル読み込み
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    # ファイルの拡張子を取得
    file_extension = os.path.splitext(path)[1]
    # ファイル名（拡張子を含む）を取得
    file_name = os.path.basename(path)

    # CSVは行単位で分割せず、1ファイルを1ドキュメントとして読み込む
    if file_extension == ".csv":
        docs = load_csv_as_single_document(path)
        docs_all.extend(docs)
        return

    # 想定していたファイル形式の場合のみ読み込む
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        # ファイルの拡張子に合ったdata loaderを使ってデータ読み込み
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()
        docs_all.extend(docs)


def load_csv_as_single_document(path):
    """
    CSVを1ファイル=1ドキュメントとして読み込む

    Args:
        path: CSVファイルパス

    Returns:
        読み込んだドキュメントのリスト
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        employee_rows = []
        by_department = {}

        for i, row in enumerate(reader, start=1):
            columns = [f"{header}:{row.get(header, '').strip()}" for header in headers]

            dept_candidates = [
                row.get(header, "").strip() for header in headers
                if "部署" in header or "所属" in header
            ]
            departments = [d for d in dept_candidates if d]
            if departments:
                columns.append(f"所属部署:{' / '.join(departments)}")

            employee_line = f"従業員{i}: " + " / ".join(columns)
            employee_rows.append(employee_line)

            for dept in departments:
                by_department.setdefault(dept, []).append(employee_line)

    department_sections = []
    for dept, rows in by_department.items():
        department_sections.append(f"{dept} に所属する従業員数: {len(rows)}")
        department_sections.extend(rows)

    content = "\n".join([
        "これは社員名簿データです。従業員情報を一覧化する質問には、該当する全員を列挙してください。",
        f"レコード件数: {len(employee_rows)}",
        "部署別一覧:",
        *department_sections,
        "全従業員一覧:",
        *employee_rows,
    ])

    return [
        LCDocument(
            page_content=content,
            metadata={"source": path, "csv_merged": True, "record_count": len(employee_rows)},
        )
    ]


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s