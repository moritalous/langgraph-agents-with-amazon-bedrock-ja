# Amazon Bedrock を使用した LangGraph エージェント

このリポジトリには、[Harrison Chase](https://www.linkedin.com/in/harrison-chase-961287118) ([LangChain](https://www.langchain.com/) の共同創設者兼 CEO) と [Rotem Weiss](https://www.linkedin.com/in/rotem-weiss) ([Tavily](https://tavily.com/) の共同創設者兼 CEO) によって作成され、[DeepLearning.AI](https://www.deeplearning.ai/) でホストされているコース [AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) から改変されたワークショップが含まれています。

元のコンテンツは、著者の同意を得て使用されています。

このワークショップは、AWS Workshop Studio [こちら](https://catalog.us-east-1.prod.workshops.aws/workshops/9bc28f51-d7c3-468b-ba41-72667f3273f1/en-US) でもご利用いただけます。

スムーズな体験を実現するために、資料をご覧になる前に必ずこの README を読んで従ってください。

## 概要

ワークショップの内容:
- 関数呼び出し LLM の改善やエージェント検索などの専用ツールを活用し、AI エージェントとエージェント ワークフローの最新の進歩を探ります
- LangChain のエージェント ワークフローに対する更新されたサポートを活用し、複雑なエージェント動作を構築するための拡張機能である LangGraph を紹介します
- *計画、ツールの使用、リフレクション、マルチエージェント通信、メモリ* を含むエージェント ワークフローの主要な設計パターンに関する洞察を提供します

資料は 6 つの Jupyter Notebooks ラボに分かれており、LangGraph フレームワーク、その基礎となる概念、および Amazon Bedrock での使用方法を理解するのに役立ちます:

- ラボ 1: [ReAct エージェントをゼロから構築する](Lab_1/)
- Python と LLM を使用して基本的な ReAct エージェントをゼロから構築し、ツールの使用と観察を通じてタスクを解決するための推論と動作のループを実装します
- ラボ 2: [LangGraph コンポーネント](Lab_2/)
- 循環グラフを使用してエージェントを実装するためのツールである LangGraph の紹介ノード、エッジ、状態管理などのコンポーネントを使用して、より構造化され制御可能なエージェントを作成する方法を示します
- ラボ 3: [エージェント検索ツール](Lab_3/)
- エージェント検索ツールの紹介。動的なソースから構造化された関連データを提供することで AI エージェントの機能を強化し、精度を向上させ、幻覚を軽減します
- ラボ 4: [永続性とストリーミング](Lab_4/)
- 永続性とストリーミングは、エージェントの長期タスクに不可欠であり、状態の保存、会話の再開、エージェントのアクションと出力のリアルタイムの可視性を可能にします
- ラボ 5: [人間が関与するループ](Lab_5/)
- LangGraph の高度な人間が関与するループ インタラクション パターン。中断の追加、状態の変更、タイム トラベル、手動状態の更新など、AI エージェントの制御とインタラクションを改善します
- ラボ 6: [エッセイ ライター](Lab_6/)
- 計画、調査、執筆、考察、改訂を含む複数のステップのプロセスを使用して AI エッセイ ライターを構築し、実装します相互接続されたエージェントのグラフとして

LangGraph を初めて使用する場合は、[オリジナル コース](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) を参照して、詳細なビデオの説明を確認することをお勧めします。

環境の設定を始めましょう。

## 仮想環境をセットアップする

この手順は、[AWS 認証](https://docs.aws.amazon.com/cli/v1/userguide/cli-authentication-short-term.html) を使用してローカルで使用すること、および [Amazon SageMaker JupyterLab](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl.html) または [Amazon SageMaker コード エディタ](https://docs.aws.amazon.com/sagemaker/latest/dg/code-editor.html) インスタンス内で使用することを目的としています。

このコースには `Python >=3.10` が必要です (インストールするには、このリンクにアクセスしてください: https://www.python.org/downloads/)

### 1. リポジトリをダウンロードします

```
git clone https://github.com/aws-samples/langgraph-agents-with-amazon-bedrock.git
```

### 2. OS 依存関係をインストールします (Ubuntu/Debian)

```
sudo apt update
sudo apt-get install graphviz graphviz-dev python3-dev
pip install pipx
pipx install poetry
pipx Ensurepath
source ~/.bashrc
```

他の OS のインストール コマンドについては、こちらを参照してください: https://pygraphviz.github.io/documentation/stable/install.html

### 3. 仮想環境を作成し、Python 依存関係をインストールします

```
cd langgraph-agents-with-amazon-bedrock
export POETRY_VIRTUALENVS_PATH="$PWD/.venv"
export INITIAL_WORKING_DIRECTORY=$(pwd)
poetry shell
```

```
cd $INITIAL_WORKING_DIRECTORY
poetry install
```

### 4. Jupyter Notebook サーバーにカーネルを追加する
新しく作成した Python 環境を、Jupyter Notebook サーバーの使用可能なカーネルのリストに追加する必要があります。
これは、poetry 環境内から次のコマンドで実行できます:
```
poetry run python -m ipykernel install --user --name agents-dev-env
```
カーネルがすぐにリストに表示されない場合があります。その場合は、リストを更新する必要があります。

### 5. Tavily API キーを作成して設定する

https://app.tavily.com/home にアクセスして、API キーを無料で作成します。

### 6. ローカル環境変数を設定する

個人情報がコミットされないように、[.gitignore](.gitignore) にすでに追加されている一時環境ファイル [env.tmp](env.tmp) の個人用コピーを `.env` という名前で作成します。
```
cp env.tmp .env
```
必要に応じて、`.env` ファイル内で Amazon Bedrock を使用するための優先リージョンを編集できます (デフォルトは `us-east-1`)。

### 7. Tavily API キーを保存する
Tavily API キーを保存するには、2 つのオプションがあります:

1. Tavily API キーを `.env` ファイル内にコピーします。このオプションは常に最初にチェックされます。

2. [AWS Secrets Manager で「TAVILY_API_KEY」という名前で新しいシークレットを作成](https://docs.aws.amazon.com/secretsmanager/latest/userguide/create_secret.html)し、シークレット `arn` をクリックして取得し、[シークレットの読み取り権限](https://docs.aws.amazon.com/secretsmanager/latest/userguide/auth-and-access_examples.html#auth-and-access_examples_read)を持つ[インラインポリシー](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_manage-attach-detach.html#add-policies-console)を [SageMaker 実行ロール](https://docs.aws.amazon.com/sagemaker/latest/dg/domain-user-profile-view-describe.html) に追加して、以下の例でコピーした `arn` を置き換えます。
```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "secretsmanager:GetSecretValue",
            "Resource": "arn:aws:secretsmanager:<Region>:<AccountId>:secret:SecretName-6RandomCharacters"
        }
    ]
}
```

準備完了です。各ノートブックに新しく作成された `agents-dev-env` カーネルを選択してください。

# 追加リソース

- [Amazon Bedrock ユーザーガイド](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)
- [LangChain ドキュメント](https://python.langchain.com/v0.2/docs/introduction/)
- [LangGraph github リポジトリ](https://github.com/langchain-ai/langgraph)
- [LangSmith Prompt ハブ](https://smith.langchain.com/hub)