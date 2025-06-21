import torch
import spacy
import scispacy
from scispacy.linking import EntityLinker
import os
import re
import pickle
from typing import Dict, Any, List, Set, Tuple
from transformers import AutoTokenizer, AutoModel

# --------------------------------------------------------------------------
# 導入 MEDIQ 框架的基礎設施和我們自定義的 KGReasoner
# --------------------------------------------------------------------------
try:
    from expert import Expert
    from helper import get_response, log_info
    from kg_reasoner import KGReasoner
except ImportError as e:
    print(f"錯誤：導入必要模組失敗。請確保 kg_expert.py 與 expert.py, helper.py, 和 kg_reasoner_predictor.py 位於同一 Python 路徑下。")
    print(f"導入錯誤訊息: {e}")
    
try:
    # 理想情況下，你可以直接導入
    from MediQ_trainer_triattn import Trainer, MediQAnnotatedDataset, CuiEmbedding 
except ImportError:
    print("警告：無法從 MediQ_trainer_triattn.py 導入 Trainer。")
    print("將使用佔位符類別，請確保將真實的 Trainer 類別及其依賴項放在此處。")

# --------------------------------------------------------------------------
# KGExpert 類別實現
# --------------------------------------------------------------------------
class KGExpert(Expert):
    """
    一個整合了知識圖譜推理的模組化、多輪互動專家系統，具備詳盡的推理日誌記錄功能。
    """
    def __init__(self,  args, inquiry: str, options: Dict[str, str]):
        args = None
        super().__init__(args,inquiry, options)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        self.kg_reasoner = self._initialize_kg_reasoner()
        self.nlp = self._initialize_scispacy()
        self.graph_cui_set = set(self.kg_reasoner.gmodel.g_tensorized['cui_to_idx'].keys()) if self.kg_reasoner and hasattr(self.kg_reasoner, 'gmodel') else set()
        
        # 初始化豐富的推理日誌結構
        self.reasoning_log = self._initialize_reasoning_log(inquiry, options)
        log_info(f"KGExpert 初始化完畢，推理日誌已創建。")

    # --- 模組 1: 初始化與狀態管理 ---
    def _initialize_kg_reasoner(self) -> KGReasoner:
        """處理 KGReasoner 的初始化。"""
        print("開始演示 KGReasoner 的使用...")

        # --- 1. 定義所有必要資源的路徑 (請替換為你的真實路徑) ---
        CUI_TO_TEXT_MAP_FILE = "./drknows/sm_t047_cui_aui_eng.pkl"
        GRAPH_NX_FILE = "./drknows/SNOMED_CUI_MAJID_Graph_wSelf.pkl"
        CUI_EMBEDDING_FILE = "./drknows/GraphModel_SNOMED_CUI_Embedding.pkl"
        TOKENIZER_PATH = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        
        # --- 你的預訓練模型權重路徑 ---
        SAVED_GMODEL_PATH = "./saved_models_mediq/gmodel_mediq_best.pth"
        SAVED_ENCODER_PATH = "./saved_models_mediq/encoder.pth"
        
        # 檢查文件是否存在
        required_files = [CUI_TO_TEXT_MAP_FILE, GRAPH_NX_FILE, CUI_EMBEDDING_FILE, SAVED_GMODEL_PATH, SAVED_ENCODER_PATH]
        for f_path in required_files:
            if not os.path.exists(f_path):
                print(f"錯誤：找不到必要的資源文件 -> {f_path}")
                print("請確保所有路徑都已正確設置，並將此 main block 中的佔位符替換掉。")
                exit()

        # --- 2. 準備 Trainer 初始化所需的參數 ---
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("正在加載基礎資源 (tokenizer, encoder, graph)...")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        base_encoder_model = AutoModel.from_pretrained(TOKENIZER_PATH)
        g_nx_loaded = pickle.load(open(GRAPH_NX_FILE, "rb"))
        cui_embedding_lookup_obj = CuiEmbedding(CUI_EMBEDDING_FILE, device=device)
        _nodes_for_vocab = sorted(list(g_nx_loaded.nodes()))
        cui_vocab_for_trainer = {cui_str: i for i, cui_str in enumerate(_nodes_for_vocab)}
        
        # Trainer 的超參數 (應與你訓練時使用的參數一致)
        hdim = base_encoder_model.config.hidden_size
        
        # 將所有參數打包到一個字典中
        trainer_kwargs = {
            'tokenizer': tokenizer,
            'encoder': base_encoder_model,
            'g_nx': g_nx_loaded,
            'cui_embedding_lookup': cui_embedding_lookup_obj,
            'hdim': hdim,
            'nums_of_head': 3,
            'cui_vocab_str_to_idx': cui_vocab_for_trainer,
            'top_n': 8,
            'device': device,
            'nums_of_epochs': 1, # 對預測不重要
            'LR': 1e-5,  # 對預測不重要
            'gnn_update': True, 
            'path_encoder_type': "Transformer",
            'path_ranker_type': "Flat",
            'gnn_type': 'GAT',   # 應與你訓練的模型匹配
            'score_threshold': 0.7,
            'gin_hidden_dim': hdim,
            'gin_num_layers': 2,
        }
        
        # --- 3. 實例化 KGReasoner ---
        try:
            kg_reasoner_instance = KGReasoner(
                cui_to_text_map_path=CUI_TO_TEXT_MAP_FILE,
                **trainer_kwargs
            )
            
            # 加載預訓練的權重
            kg_reasoner_instance.gmodel.load_state_dict(torch.load(SAVED_GMODEL_PATH, map_location=device))
            kg_reasoner_instance.encoder.load_state_dict(torch.load(SAVED_ENCODER_PATH, map_location=device))
            print("成功加載預訓練模型權重。")

        except Exception as e:
            print(f"實例化或加載 KGReasoner 時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            exit()
            
        return kg_reasoner_instance
        

    def _initialize_scispacy(self) -> object:
        """初始化 ScispaCy 模型。"""
        try:
            nlp = spacy.load("en_core_sci_md")
            linker = EntityLinker(resolve_abbreviations=True, name="umls")
            nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
            print("成功加載 ScispaCy 模型。")
            return nlp
        except Exception as e:
            print(f"警告：ScispaCy 模型加載失敗: {e}。")
            return None

    def _initialize_reasoning_log(self, inquiry: str, options: Dict) -> Dict:
        """創建並初始化推理日誌的結構。"""
        task_description = '''
        This is an interactive clinical reasoning task aiming to find the most likely diagnosis through multi-turn dialogue.
        The objective is to answer a multiple-choice question, but only limited patient information is provided at the start. A premature answer might overlook important unknown information and lead to misjudgment. Our system should attempt to acquire more patient information before answering.
        Our system consists of the following modules:
        1. Knowledge Graph Reasoning: Based on patient-provided information, it reasons within a reliable knowledge graph to provide information suggestions represented as paths.
        2. Path Analysis: It analyzes the paths obtained from reasoning to generate a more readable core context.
        3. Intermediate Answer Generation: It generates an intermediate answer based on the analyzed core context.
        4. Abstention Decision: It makes an abstention decision based on the intermediate answer and patient-provided information.
        5. Question Generation: It analyzes the currently required information and generates a question.
        '''
        return {
            "session_preamble": {
                "task_description": task_description,
                "system_workflow": "The system will iteratively execute: 1. Suggest Focus CUIs -> 2. KG Reasoning -> 3. Path Analysis -> 4. Intermediate Answer Generation -> 5. Abstention Decision -> 6. Action (Answer/Question).",
                "main_inquiry": inquiry,
                "options": options,
            },
            "turns": []
        }

    def _format_log_for_prompt(self, current_turn_data: Dict = None) -> str:
        """
        Formats the structured reasoning log into a human-readable string for the LLM.
        """
        log_text = ""
        log_text += "=== SYSTEM WORKFLOW & LOG DEFINITIONS ===\n"
        preamble = self.reasoning_log['session_preamble']
        log_text += f"[Task Description]: {preamble['task_description']}\n"
        log_text += f"[System Workflow]: {preamble['system_workflow']}\n"
        log_text += "[Log Components Definition]:\n"
        log_text += "- Reasoning Focus: The starting concepts (CUIs) for this turn's KG reasoning.\n"
        log_text += "- Core Context: A clinical insight summary of the KG reasoning paths.\n"
        log_text += "- Guess for this Turn: A preliminary guess for the final question based on this turn's information.\n"
        log_text += "- Confidence in Answering: A 5-level scale assessing the confidence to answer the final question directly.\n"
        log_text += "- Question Asked: The clarifying question posed to the patient if confidence is low.\n"
        log_text += "- Patient's Response: The patient's actual response to the previous question.\n"

        log_text += "\n=== DIAGNOSTIC REASONING LOG ===\n"
        log_text += f"[Task Objective]: {preamble['main_inquiry']}\n"
        log_text += f"[Initial Patient Info]: {preamble.get('patient_initial_info', 'N/A')}\n"
        options_text = " ".join([f"{k}:{v}" for k, v in preamble['options'].items()])
        log_text += f"[Diagnostic Options]: {options_text}\n"
        log_text += "--------------------\n"

        for i, turn in enumerate(self.reasoning_log['turns']):
            log_text += f"--- Turn {i+1} (Completed) ---\n"
            focus_module = turn.get('1_focus_cuis_module', {})
            # Convert CUI codes to readable strings
            focus_cuis_readable = [f"'{self.kg_reasoner.cui_to_text_map.get(cui, cui)}' ({cui})" for cui in focus_module.get('cuis', [])]
            log_text += f"[Reasoning Focus]: {', '.join(focus_cuis_readable)}\n  - (Reason: {focus_module.get('reasoning', 'N/A')})\n"
            log_text += f"[Core Context]: {turn.get('3_path_analysis_module', {}).get('core_context', 'None')}\n"
            log_text += f"[Guess for this Turn]: {turn.get('4_intermediate_choice_module', {}).get('choice', 'N/A')}\n"
            abstention_module = turn.get('5_abstention_module', {})
            log_text += f"[Confidence in Answering]: {abstention_module.get('decision', {}).get('rating', 'N/A')}\n  - (Reason: {abstention_module.get('reasoning', 'N/A')})\n"
            action_module = turn.get('6_action_module', {})
            if action_module.get('type') == 'question':
                log_text += f"[Question Asked]: {action_module.get('content')}\n"
                log_text += f"[Patient's Response]: {turn.get('7_patient_response', 'Awaiting response')}\n"
            else:
                log_text += f"[Final Decision]: Answered with option {action_module.get('content')}\n"
            log_text += "--------------------\n"
        
        if current_turn_data:
            log_text += f"--- Turn {current_turn_data['turn_number']} (In Progress) ---\n"
        
        return log_text

    # --- 模組 7 & 1: 焦點建議與 CUI 提取 ---
    def _update_focus_cuis(self, formatted_log: str, turn_data: Dict) -> Tuple[str, List[str]]:
        """
        (步驟 1 & 7) 更新下一輪推理的焦點 CUIs。
        後續輪次將由 LLM 生成其選擇新焦點的理由。
        """
        turn_number = len(self.reasoning_log['turns']) + 1
        
        if turn_number == 1:
            # 第一回合的邏輯保持不變，使用模板化理由
            reasoning = "Turn 1: Extracting all valid CUIs from the patient's initial info as the starting point."
            initial_info = self.reasoning_log['session_preamble'].get('patient_initial_info', '')
            focus_cuis = self._extract_cuis_from_text(initial_info)
        else:
            # 後續輪次，由 LLM 進行評估、更新，並生成理由
            
            # 1. 準備 Prompt 所需的變量
            
            last_turn_log = self.reasoning_log['turns'][-1]
            last_focus_cuis = last_turn_log.get('1_focus_cuis_module', {}).get('cuis', [])
            last_focus_cuis_text = ", ".join([f"'{self.kg_reasoner.cui_to_text_map.get(cui, cui)}' ({cui})" for cui in last_focus_cuis])
            

            # 2. 構建新的、引導 LLM 輸出理由的 Prompt
            prompt = f"""
            {formatted_log}

            [Your Current Task]
            You are a lead physician directing a complex diagnosis, and your task is to dynamically adjust the team's reasoning focus based on the latest patient feedback.
            You are about to start reasoning for "Turn {turn_number}". We have just received the patient's response to the last question.
            
            Our reasoning focus in the "previous turn" was based on the following concepts:
            - {last_focus_cuis_text}

            Please follow these thinking steps:
            1.  Evaluate Old Focus vs. New Information: Considering the latest patient response, which of the previous focus concepts are still important? Which have become secondary or can be ignored? What new, critical concepts has the new response revealed?
            2.  Articulate Reason for Update: Summarize your reasoning for updating the focus.
            3.  Define New Focus: Describe the new, updated clinical focus for the next round of reasoning in natural language.

            Please strictly follow this format for your output:
            REASON: [Fill in your thought process from step 2 here, explaining why you are choosing the following focus.]
            FOCUS_DESCRIPTION: [Fill in the natural language description of the new focus from step 3 here, which will be used for the next round of reasoning.]
            """
            
            messages = [{"role": "system", "content": "You are a lead clinical reasoning strategist."}, {"role": "user", "content": prompt}]
            
            # 3. 執行 LLM 調用並解析結果
            response_text, _, _ = get_response(messages, model_name="gpt-4o", temperature=0.2)
            
            reason_match = re.search(r"REASON:\s*(.*)", response_text, re.DOTALL)
            focus_match = re.search(r"FOCUS_DESCRIPTION:\s*(.*)", response_text, re.DOTALL)

            # 解析出的理由將被用於日誌記錄
            reasoning = reason_match.group(1).strip() if reason_match else "Failed to parse the reasoning for the focus suggestion."
            
            # 解析出的焦點描述將被用於提取 CUI
            focus_text = focus_match.group(1).strip() if focus_match else ""
            focus_cuis = self._extract_cuis_from_text(focus_text)

            print('prompt for focus update', prompt)
        print(f"回合 {turn_number} 的推理焦點已更新為 ({len(focus_cuis)}個): {focus_cuis}")
        print(f"焦點更新理由: {reasoning}")
        return reasoning, focus_cuis

    def _extract_cuis_from_text(self, text: str) -> List[str]:
        # (此方法實現與上一輪討論相同)
        if not self.nlp or not self.graph_cui_set: return []
        valid_cuis = []
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent._.kb_ents:
                for cui_candidate, _ in ent._.kb_ents:
                    if cui_candidate in self.graph_cui_set:
                        valid_cuis.append(cui_candidate)
                        break
        return list(set(valid_cuis)) # 返回唯一的CUI

    

    def respond(self, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        
        # 在 respond 方法開始時，先記錄 patient_initial_info (如果尚未記錄)
        if 'patient_initial_info' not in self.reasoning_log['session_preamble']:
            self.reasoning_log['session_preamble']['patient_initial_info'] = patient_state['initial_info']
            
            
        if self.reasoning_log['turns'] and patient_state['interaction_history']:
            last_log_turn = self.reasoning_log['turns'][-1]
            last_history_qa = patient_state['interaction_history'][-1]
            last_logged_question = last_log_turn.get('6_action_module', {}).get('content')
            if last_logged_question == last_history_qa['question']:
                last_log_turn['7_patient_response'] = last_history_qa['answer']
                log_info(f"日誌已更新：將回答 '{last_history_qa['answer']}' 補入回合 {last_log_turn['turn_number']}。")

        # 創建當前回合的日誌條目
        turn_number = len(self.reasoning_log['turns']) + 1
        current_turn_data = {"turn_number": turn_number}
        
        # --- 步驟 1 & 7: 更新焦點 ---
        formatted_log = self._format_log_for_prompt(current_turn_data)
        focus_reasoning, focus_cuis = self._update_focus_cuis(formatted_log, current_turn_data)
        current_turn_data['1_focus_cuis_module'] = {'reasoning': focus_reasoning, 'cuis': focus_cuis}
        
        
        # --- 步驟 2: KG 推理 ---
        predicted_paths, top_score = self.kg_reasoner.predict(focus_cuis)
        current_turn_data['2_kg_reasoner_module'] = {'output': {'top_paths': predicted_paths, 'top_score': top_score}}
        
        
        # --- 步驟 3: 路徑分析 -> 核心上下文 ---
        formatted_log = self._format_log_for_prompt(current_turn_data)
        core_context = self._analyze_and_summarize_paths(formatted_log, current_turn_data)
        current_turn_data['3_path_analysis_module'] = { 'core_context': core_context}
        
        # --- 步驟 4: 生成中間答案 ---
        formatted_log = self._format_log_for_prompt(current_turn_data)
        intermediate_reasoning, intermediate_choice = self._generate_intermediate_choice(formatted_log, current_turn_data)
        current_turn_data['4_intermediate_choice_module'] = {'reasoning': intermediate_reasoning, 'choice': intermediate_choice}
        
        # --- 步驟 5: 棄權決策 ---
        formatted_log = self._format_log_for_prompt(current_turn_data)
        abstain, abstention_reasoning, conf_score = self._decide_abstention(formatted_log, current_turn_data)
        current_turn_data['5_abstention_module'] = {'reasoning': abstention_reasoning, 'decision': {'action': 'ASK' if abstain else 'ANSWER', 'rating': conf_score}}
        
        # 步驟 6: 執行最終動作
        if not abstain:
            final_choice = intermediate_choice
            current_turn_data['6_action_module'] = {'type': 'choice', 'reasoning': 'Have enough confidence to answer', 'content': final_choice}
            self.reasoning_log['turns'].append(current_turn_data)
            return {"type": "choice", "letter_choice": final_choice}
        else:
            question_reasoning, question = self._generate_question(current_turn_data) 
            current_turn_data['6_action_module'] = {'type': 'question', 'reasoning': question_reasoning, 'content': question}
            self.reasoning_log['turns'].append(current_turn_data)
            return {"type": "question", "question": question, "letter_choice": intermediate_choice}


    def record_patient_response(self, response_text: str):
        """在主循環獲取病患回答後，調用此方法來更新最後一輪的日誌。"""
        if self.reasoning_log['turns']:
            self.reasoning_log['turns'][-1]['7_patient_response'] = response_text
            log_info(f"已記錄病患對回合 {self.reasoning_log['turns'][-1]['turn_number']} 的回答。")

    def _analyze_and_summarize_paths(self, formatted_log: str, current_turn_data: Dict) -> str:
        
        kg_output = current_turn_data.get('2_kg_reasoner_module', {}).get('output', {})
        predicted_paths = kg_output.get('top_paths', [])
        
        if not predicted_paths:
            return "The Knowledge Graph found no clear reasoning direction under the current focus."


        turn_number = current_turn_data['turn_number']

        focus_cuis = current_turn_data.get('1_focus_cuis_module', {}).get('cuis', [])
        focus_cuis_text = ", ".join([f"'{self.kg_reasoner.cui_to_text_map.get(cui, cui)}'" for cui in focus_cuis])

        formatted_paths_list = "\n".join([f"- Path: {p['path_text']} (Score: {p['score']:.2f})" for p in predicted_paths])

        prompt = f"""
        {formatted_log}
         
        [Your Current Task]
        You are at the "Path Analysis" stage of "Turn {turn_number}".
        You are a senior clinical data analyst, and your task is to translate the output of the underlying Knowledge Graph (KG) reasoning engine into a concise, precise, and clinically insightful "Core Context".

        This turn's reasoning was initiated based on the following "Focus Concepts":
        - {focus_cuis_text}

        Based on this focus, the reasoning engine found the following most relevant paths in the knowledge graph. These paths represent potential connections from known information to other possibly related unknown information:
        ---
        {formatted_paths_list}
        ---

        Please analyze the paths above, extract their clinical value, and generate the "Core Context". This context should summarize the shared clinical significance or the most important diagnostic direction these paths point to. It will serve as a crucial basis for all subsequent decisions in this turn.

        [Core Context]:
        """
        
        messages = [{"role": "system", "content": "You are a senior biomedical data analyst."}, {"role": "user", "content": prompt}]
        
        # 執行 LLM 調用
        core_context, _, _ = get_response(messages, model_name="gpt-4o", temperature=0.1)
        print('prompt for core_context',prompt)
        print(f"生成的核心上下文: {core_context}")
        return core_context

   

    def _decide_abstention(self, formatted_log: str, turn_data: Dict) -> Tuple[bool, str, int, List[Dict]]:
        

        # 1. 從當前回合數據中提取所需信息
        turn_number = turn_data['turn_number']
        core_context = turn_data.get('3_path_analysis_module', {}).get('core_context', '無')
        answer_reason = turn_data.get('4_intermediate_choice_module', {}).get('reasoning', '無')
        
        # 2. 構建包含完整上下文和明確任務的 Prompt
        prompt = f"""
        {formatted_log}
         
        [Your Current Task]
        You are at the "Confidence in Answering" evaluation stage for "Turn {turn_number}".
        You are a clinical strategist, and your task is to assess the completeness of the current diagnostic process and decide whether to make a final diagnosis or ask for more information.
        
        The Core Context for this turn is: {core_context}
        The reasoning for the guessed answer is: {answer_reason}

        Based only on the information above, determine if we have enough confidence to make a final, conclusive diagnosis. Please use the following five-level scale to express your confidence:
        "Very Confident": All evidence points to a clear option and is sufficient to rule out the others; no more information is needed.
        "Somewhat Confident": There is enough information to indicate that one option is more likely, but more information would be helpful for a conclusive decision.
        "Neither Confident or Unconfident": There is some supporting evidence, but it's unclear which option is correct.
        "Somewhat Unconfident": The evidence points to more than one option, requiring more questions for differentiation.
        "Very Unconfident": There is not enough evidence to support any of the options; the likelihood of picking the correct option is close to random guessing.

        Please strictly adhere to the following format for your output:
        REASON: [Explain here why you are confident or not, based on the core context or historical log.]
        DECISION: [Enter your chosen confidence rating from the list above.]
        """
        
        messages = [{"role": "system", "content": "You are a clinical strategist evaluating diagnostic confidence."}, {"role": "user", "content": prompt}]
        
        # 3. 執行 LLM 調用
        response_text, _, _ = get_response(messages, model_name="gpt-4o", temperature=0.1)
        
        # 4. 解析 LLM 的結構化回應
        reason_match = re.search(r"REASON:\s*(.*)", response_text, re.DOTALL)
        decision_match = re.search(r"DECISION:\s*\"?(.+?)\"?$", response_text, re.MULTILINE)
        
        reasoning_for_log = reason_match.group(1).strip() if reason_match else "Failed to parse decision reasoning."
        decision_text = decision_match.group(1).strip().lower() if decision_match else ""
        
        # 將文字評級轉換為數字分數
        conf_score = 0
        if "very confident" in decision_text: conf_score = 5
        elif "somewhat confident" in decision_text: conf_score = 4
        elif "neither" in decision_text: conf_score = 3
        elif "somewhat unconfident" in decision_text: conf_score = 2
        elif "very unconfident" in decision_text: conf_score = 1

        # 根據分數決定是否棄權（提問）
        abstain = conf_score < 6.0
        
        print('prompt for decision:', prompt)
        print(f"棄權模組理由: {reasoning_for_log}")
        print(f"棄權模組決策: '{decision_text}' -> 分數: {conf_score} -> 棄權: {abstain}")
        
        
        
        return abstain, reasoning_for_log, conf_score

    def _generate_question(self, current_turn_data: Dict) -> Tuple[str, str]:
        
        # 1. 從當前回合數據和日誌中提取所需信息
        formatted_log = self._format_log_for_prompt(current_turn_data)
        turn_number = current_turn_data['turn_number']
        core_context = current_turn_data.get('3_path_analysis_module', {}).get('core_context', '無')
        
        conf_score = current_turn_data.get('5_abstention_module', {}).get('decision', {}).get('rating', 0)
        abstention_reason = current_turn_data.get('5_abstention_module', {}).get('reasoning', "")
        
        # 獲取所有診斷選項
        options = self.reasoning_log['session_preamble']['options']
        options_text = "\n".join([f"- {k}: {v}" for k, v in options.items()])

        # 2. 構建包含鑑別診斷思維的 Prompt
        prompt = f"""      
        {formatted_log}

        [Your Current Task]
        You are at the "Question Generation" stage for "Turn {turn_number}".
        You are a senior clinical diagnostician, and your task is to ask a key question that can best differentiate between the possibilities based on the available information.

        [Core Context]
        {core_context}
        [Confidence Assessment]
        {abstention_reason}
        [Differential Diagnosis Options]
        {options_text}

        Please follow these thinking steps to construct your question:
        1.  Analysis: Based on the diagnostic reasoning log, which option is the most likely diagnosis right now? Which are secondary but still plausible differential diagnoses?
        2.  Find the Differentiating Point: Think about what single piece of key information (e.g., a symptom, a past medical history item, a lab result) would most effectively help you differentiate between the most likely diagnosis and the other options.
        3.  Construct the Question: Design a specific, atomic, clarifying question based on this "differentiating point".

        Please strictly adhere to the following format for your output:
        REASON: [Fill in your thought process from step 2 here, explaining why asking this question is effective for differential diagnosis.]
        QUESTION: [Fill in the final question you constructed from step 3 here.]
        """
        
        messages = [{"role": "system", "content": "You are a senior clinical diagnostician."}, {"role": "user", "content": prompt}]
        
        # 3. 執行 LLM 調用並解析結果
        response_text, _, _ = get_response(messages, model_name='gpt-4o')
        
        # 解析 REASON 和 QUESTION
        reason_match = re.search(r"REASON:\s*(.*)", response_text, re.DOTALL)
        question_match = re.search(r"QUESTION:\s*(.*)", response_text, re.DOTALL)
        
        reasoning_for_log = reason_match.group(1).strip() if reason_match else "Failed to parse the reasoning for the question."
        final_question = question_match.group(1).strip() if question_match else "Could you provide more details about your symptoms?" # Fallback

        print('prompt for question generation:', prompt)
        print(f"提問理由: {reasoning_for_log}")
        print(f"生成的澄清問題: {final_question}")
        
        return reasoning_for_log, final_question

    def _generate_intermediate_choice(self, formatted_log: str, turn_data: Dict) -> Tuple[str, str]:

        # 1. 獲取當前回合編號
        turn_number = turn_data['turn_number']
        core_context = turn_data.get('3_path_analysis_module', {}).get('core_context', 'N/A')
        
        # 2. 構建包含完整上下文的 Prompt
        prompt = f"""
        {formatted_log}
        
        [Your Current Task]
        You are at the "Final Decision" stage for "Turn {turn_number}".
        You are a senior clinical diagnostician making a final conclusion. This is the last step of the diagnostic process.
        
        The Core Context for this turn is: {core_context}
        Please make a final, conclusive diagnostic choice based on the entire reasoning process above.

        Please strictly adhere to the following format for your output:
        REASON: [Summarize the key clinical reasons for your final choice here.]
        FINAL_CHOICE: [Enter the final answer letter (A, B, C, or D) here.]
        """
        
        messages = [{"role": "system", "content": "You are a senior clinical diagnostician making a final conclusion."}, {"role": "user", "content": prompt}]
        
        # 3. 執行 LLM 調用並解析結果
        response_text, _, _ = get_response(messages, model_name="gpt-4o", temperature=0.0) # 使用零溫度以求最確定的輸出
        
        # 解析 REASON 和 FINAL_CHOICE
        reason_match = re.search(r"REASON:\s*(.*)", response_text, re.DOTALL)
        choice_match = re.search(r"FINAL_CHOICE:\s*([A-D])", response_text, re.DOTALL)
        
        reasoning_for_log = reason_match.group(1).strip() if reason_match else "Failed to parse the final reasoning."
        final_choice = choice_match.group(1).strip().upper() if choice_match else "A" # Fallback

        # 確保 fallback 的選項是有效的
        if final_choice not in self.options:
            final_choice = "A"

        print('prompt for final choice:', prompt)
        print(f"最終決策理由: {reasoning_for_log}")
        print(f"最終診斷選項: {final_choice}")
        
        return reasoning_for_log, final_choice
    
    
if __name__ == '__main__':
    
    print("="*30)
    print(" KGExpert 互動測試模式 ")
    print("="*30)
    print("您將扮演病患的角色，請根據專家系統的問題輸入您的回答。")

    # 1. 準備測試案例和參數
    test_sample = {
        "id": 0,
        "question": "The mechanism of action of the medication given blocks cell wall synthesis, which of the following was given?",
        "context": [
            "A 21-year-old sexually active male complains of fever, pain during urination, and inflammation and pain in the right knee.",
            #"A culture of the joint fluid shows a bacteria that does not ferment maltose and has no polysaccharide capsule.",
            #"The physician orders antibiotic therapy for the patient."
        ],
        "options": {"A": "Gentamicin", "B": "Ciprofloxacin", "C": "Ceftriaxone", "D": "Trimethoprim"},
        "answer": "Ceftriaxone",
        "answer_idx": "C"
    }

    

    # 2. 初始化 KGExpert
    expert = KGExpert(
        inquiry=test_sample["question"],
        options=test_sample["options"]
    )

    # 3. 準備初始的 patient_state
    patient_state = {
        "initial_info": " ".join(test_sample["context"]),
        "interaction_history": []
    }
    
    max_turns = 5
    for turn in range(1, max_turns + 1):
        print(f"\n----------- 回合 {turn} -----------")
        
        # 4. 調用專家系統
        response_dict = expert.respond(patient_state)
        
        # 5. 檢查專家的決策
        if response_dict["type"] == "choice":
            print(f"\n[專家系統做出最終診斷]")
            print(f"> 診斷結果: 選項 {response_dict['letter_choice']}")
            print(f"> 正確答案: 選項 {test_sample['answer_idx']} ({test_sample['answer']})")
            if response_dict['letter_choice'] == test_sample['answer_idx']:
                print(">>> 診斷正確！")
            else:
                print(">>> 診斷錯誤。")
            break
        
        elif response_dict["type"] == "question":
            expert_question = response_dict["question"]
            print(f"專家系統提問: {expert_question}")
            
            # 6. 獲取您的輸入作為病患的回答
            patient_answer = input("您的回答 (扮演病患): ")
            
            # 7. 更新對話歷史和日誌
            patient_state["interaction_history"].append({
                "question": expert_question,
                "answer": patient_answer
            })
            expert.record_patient_response(patient_answer)
            
            if turn == max_turns:
                print("\n已達到最大對話輪數，測試結束。")