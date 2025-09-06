import streamlit as st
import pandas as pd
from joblib import load
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
import io

# --- Configuração da Página ---
st.set_page_config(
    page_title="Predição de Biomarcadores AD",
    page_icon="🧠",
    layout="wide"
)

# --- Funções de Carregamento de Modelos (Otimizadas com Cache) ---
@st.cache_resource
def load_full_pipeline(save_prefix):
    """Carrega um pipeline completo (imputer, preprocessor, model)."""
    try:
        imputer = load(f"{save_prefix}_imputer.joblib")
        preprocessor = load(f"{save_prefix}_preprocessor.joblib")
        model = load(f"{save_prefix}_model.joblib")
        return imputer, preprocessor, model
    except FileNotFoundError:
        st.error(f"Erro: Arquivos de modelo para o prefixo '{save_prefix}' não encontrados.")
        return None, None, None

@st.cache_resource
def load_stacking_pipeline(save_prefix):
    """Carrega um pipeline de stacking."""
    try:
        pipeline = load(f"{save_prefix}_stacking_pipeline.joblib")
        return pipeline
    except FileNotFoundError:
        st.error(f"Erro: Arquivo do pipeline de stacking '{save_prefix}' não encontrado.")
        return None

# --- Função de Predição e Explicação ---
def predict_and_explain(new_data, save_prefix, model_type='model'):
    """
    Realiza a predição e prepara os dados para a explicação SHAP.
    Retorna um dicionário com predição, probabilidade, modelo e dados para o SHAP.
    """
    if model_type == 'stacking':
        pipeline = load_stacking_pipeline(save_prefix)
        if pipeline is None: return None
        
        try:
            imputer = load("a_xgboost_imputer.joblib") 
            imputed_data = pd.DataFrame(imputer.transform(new_data), columns=new_data.columns)
        except FileNotFoundError:
            st.error("Imputer padrão ('a_xgboost_imputer.joblib') não encontrado. A predição do Stacking pode falhar se houver dados ausentes.")
            imputed_data = new_data 

        preds = pipeline.predict(imputed_data)
        proba = pipeline.predict_proba(imputed_data)[:, 1]
        
        return {
            "prediction": preds[0],
            "probability": proba[0],
            "explainer_model": pipeline,
            "processed_data": imputed_data,
        }

    else: # Modelo padrão (XGBoost, RandomForest)
        imputer, preprocessor, model = load_full_pipeline(save_prefix)
        if model is None: return None

        X_imp = pd.DataFrame(imputer.transform(new_data), columns=new_data.columns)
        X_proc = preprocessor.transform(X_imp)
        
        preds = model.predict(X_proc)
        proba = model.predict_proba(X_proc)[:, 1]

        try:
            raw_feature_names = preprocessor.get_feature_names_out(input_features=new_data.columns)
            cleaned_feature_names = [name.split('__')[-1] for name in raw_feature_names]
        except AttributeError: 
            cleaned_feature_names = new_data.columns.tolist()
            
        X_proc_df = pd.DataFrame(X_proc.toarray() if hasattr(X_proc, 'toarray') else X_proc, columns=cleaned_feature_names)

        return {
            "prediction": preds[0],
            "probability": proba[0],
            "explainer_model": model,
            "processed_data": X_proc_df,
        }

# --- FUNÇÃO DE PLOTAGEM (VERSÃO FINAL E ROBUSTA) ---
def plot_shap_waterfall(model, data_for_shap):
    """
    Gera e exibe um gráfico waterfall do SHAP de forma robusta,
    renderizando para um buffer de memória para evitar bugs do backend do matplotlib.
    """
    st.info("Calculando a explicação do modelo... Isso pode levar um momento.")
    
    if isinstance(model, Pipeline):
        predict_fn = lambda df: model.predict_proba(df)[:, 1]
        explainer = shap.PermutationExplainer(predict_fn, data_for_shap)
        shap_values = explainer(data_for_shap)
    else: 
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(data_for_shap)

    st.markdown("##### Análise de Influência (SHAP Waterfall Plot)")
    st.write("O gráfico abaixo mostra as características que mais empurraram a predição para 'Positivo' (vermelho) ou 'Negativo' (azul).")

    # Criar a figura é importante para ter controle
    fig, ax = plt.subplots()

    sv_instance = shap_values[0]
    if sv_instance.values.ndim > 1:
        sv_instance = sv_instance[:, 1]

    explanation_for_plot = shap.Explanation(
        values=sv_instance.values,
        base_values=sv_instance.base_values,
        data=sv_instance.data,
        feature_names=data_for_shap.columns.tolist()
    )

    shap.waterfall_plot(explanation_for_plot, max_display=15, show=False)

    # **A CORREÇÃO DEFINITIVA: RENDERIZAR PARA UM BUFFER DE MEMÓRIA**
    # 1. Criar um buffer de bytes na memória.
    buf = io.BytesIO()

    # 2. Salvar a figura no buffer. `bbox_inches='tight'` é uma alternativa mais robusta
    #    ao `tight_layout` ou `subplots_adjust` e força o cálculo correto dos limites.
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)

    # 3. Exibir a imagem diretamente a partir do buffer, em vez de passar o objeto fig.
    st.image(buf)

    # 4. Fechar a figura para liberar memória e evitar que ela seja exibida novamente.
    plt.close(fig)


# --- Constantes e Configurações ---
FEATURES = [
    'SEX_F', 'MOCA', 'EDUCATION', 'BMI', 'DIAGNOSIS_CI', 'AGE',
    'HYPERTENSION', 'HYPERLIPIDEMIA', 'HEART', 'TBI', 'DM', 'STROKE'
]

TARGETS = {
    "A (Amiloide)": ("a_xgboost", "model"),
    "T (Tau)": ("t_random_forest", "model"),
    "AT (Stacking Amiloide+Tau)": ("at_stacking", "stacking"),
}

FRIENDLY_NAMES = {
    'SEX_F': 'Sexo (Feminino)', 'MOCA': 'MoCA (Avaliação Cognitiva)', 
    'EDUCATION': 'Anos de Educação', 'BMI': 'IMC (Índice de Massa Corporal)',
    'DIAGNOSIS_CI': 'Diagnóstico de Comp. Cognitivo', 'AGE': 'Idade',
    'HYPERTENSION': 'Hipertensão', 'HYPERLIPIDEMIA': 'Hiperlipidemia',
    'HEART': 'Doença Cardíaca', 'TBI': 'Lesão Cerebral Traumática',
    'DM': 'Diabetes Mellitus', 'STROKE': 'AVC (Derrame)'
}

# --- Interface do Streamlit ---
st.title("🧠 Predição de Biomarcadores para Doença de Alzheimer")
st.markdown("Esta ferramenta utiliza modelos de Machine Learning para prever a positividade dos biomarcadores Amiloide (A) e Tau (T) com base em dados clínicos.")

st.sidebar.header("Configurações da Predição")
target_label = st.sidebar.selectbox("🎯 Escolha o Alvo:", list(TARGETS.keys()))
save_prefix, model_type = TARGETS[target_label]

input_mode = st.sidebar.radio("📊 Modo de Entrada de Dados:", ("Formulário Manual", "Upload de CSV"), horizontal=True)

st.divider()

if input_mode == "Formulário Manual":
    st.header("📋 Preencha os Dados do Paciente")
    
    with st.form("manual_input_form"):
        user_input = {}
        col1, col2, col3 = st.columns(3)
        cols_map = {0: col1, 1: col2, 2: col3}
        
        for i, feat in enumerate(FEATURES):
            with cols_map[i % 3]:
                if feat in ['SEX_F', 'DIAGNOSIS_CI', 'HYPERTENSION', 'HYPERLIPIDEMIA', 'HEART', 'TBI', 'DM', 'STROKE']:
                    user_input[feat] = st.selectbox(
                        FRIENDLY_NAMES.get(feat, feat), 
                        options=[0, 1], 
                        format_func=lambda x: "Sim" if x == 1 else "Não",
                        key=feat
                    )
                elif feat == 'AGE':
                     user_input[feat] = st.number_input(FRIENDLY_NAMES.get(feat, feat), min_value=40, max_value=120, value=65, step=1, key=feat)
                elif feat == 'MOCA':
                     user_input[feat] = st.number_input(FRIENDLY_NAMES.get(feat, feat), min_value=0.0, max_value=30.0, value=26.0, step=0.5, key=feat)
                else:
                    user_input[feat] = st.number_input(FRIENDLY_NAMES.get(feat, feat), value=0.0, step=0.1, key=feat)
        
        submitted = st.form_submit_button("Realizar Predição", type="primary", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([user_input])
        with st.spinner("Analisando os dados e gerando a predição..."):
            result = predict_and_explain(input_df, save_prefix, model_type)
        
        if result:
            st.subheader("Resultados da Análise")
            pred_col, prob_col = st.columns(2)
            
            with pred_col:
                pred_label = "Positivo" if result["prediction"] == 1 else "Negativo"
                st.metric(label=f"Predição para o Alvo '{target_label}'", value=pred_label)

            with prob_col:
                st.metric(label="Probabilidade de ser Positivo", value=f"{result['probability']:.2%}")

            st.progress(float(result['probability']), text=f"{result['probability']:.2%} de chance")

            st.divider()

            with st.expander("🔍 Ver Explicação Detalhada da Predição (SHAP)"):
                 plot_shap_waterfall(result["explainer_model"], result["processed_data"])

else: # Modo Upload CSV
    st.header("⬆️ Envio de Arquivo CSV para Predição em Lote")
    st.info("O arquivo CSV deve conter exatamente as seguintes colunas, na ordem que preferir:")
    st.code(", ".join(FEATURES))
    
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if set(FEATURES).issubset(df.columns):
                st.success("Arquivo CSV carregado com sucesso!")
                
                if st.button("Realizar Predição em Lote", type="primary", use_container_width=True):
                    with st.spinner("Processando o arquivo..."):
                        all_preds = []
                        all_probas = []

                        if model_type == 'stacking':
                            pipeline = load_stacking_pipeline(save_prefix)
                            imputer = load("a_xgboost_imputer.joblib")
                            if pipeline and imputer:
                                imputed_df = pd.DataFrame(imputer.transform(df[FEATURES]), columns=FEATURES)
                                all_preds = pipeline.predict(imputed_df)
                                all_probas = pipeline.predict_proba(imputed_df)[:, 1]
                        else:
                            imputer, preprocessor, model = load_full_pipeline(save_prefix)
                            if imputer and preprocessor and model:
                                X_imp = pd.DataFrame(imputer.transform(df[FEATURES]), columns=FEATURES)
                                X_proc = preprocessor.transform(X_imp)
                                all_preds = model.predict(X_proc)
                                all_probas = model.predict_proba(X_proc)[:, 1]

                        result_df = df.copy()
                        result_df[f"Predicao_{target_label}"] = all_preds
                        result_df[f"Probabilidade_{target_label}"] = all_probas
                        
                        st.dataframe(result_df)
                        
                        csv_data = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Baixar Resultados em CSV",
                            data=csv_data,
                            file_name=f"resultados_{save_prefix}.csv",
                            mime="text/csv",
                        )
            else:
                missing_cols = set(FEATURES) - set(df.columns)
                st.error(f"Erro: O arquivo CSV não contém todas as colunas necessárias. Estão faltando: {', '.join(missing_cols)}")
        except Exception as e:
            st.error(f"Ocorreu um erro ao processar o arquivo: {e}")

st.sidebar.divider()
st.sidebar.info("Modelos baseados em pipelines de HPO e explicabilidade com SHAP.")
