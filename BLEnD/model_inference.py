from utils import *
import sys
import os
import json
import json_repair

# Add evaluation directory to path for persona utilities
eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation')
if eval_dir not in sys.path:
    sys.path.append(eval_dir)
from persona_util import persona_prompt_saq, persona_refine_prompt_saq

parser = argparse.ArgumentParser(description='Choose your model(s) & language(s)')
parser.add_argument('--model',type=str,
                    help='Provide the model you want to use. Check and choose from the key values of the MODEL_PATHS variable. If you want to test on multiple models, provide multiple model names with ", " between each (e.g., "gpt-4-0125-preview, aya-101").')
parser.add_argument('--language',type=str,default=None,
                    help='Provide the language you want to test on. Check and choose from the first values of the LANG_COUNTRY variable. If you want to test on multiple languages, provide multiple languages with ", " between each (e.g., "English, Korean").')
parser.add_argument('--country',type=str,default=None,
                    help='Provide the country you want to test on. Check and choose from the second values of the LANG_COUNTRY variable. If you want to test on multiple countries, provide multiple countries with ", " between each (e.g., "UK, South Korea"). Make sure you have the same number of countries and languages provided. The language-country pair do not have to be identical with the pairs within the LANG_COUNTRY variable.')
parser.add_argument('--question_dir',type=str,default=None,
                    help='Provide the directory name with (translated) questions.')
parser.add_argument('--question_file',type=str,default=None,
                    help='Provide the csv file name with (translated) questions.')
parser.add_argument('--question_col',type=str,default=None,
                    help='Provide the column name from the given csv file name with (translated) questions.')
parser.add_argument('--id_col',type=str,default="ID",
                    help='Provide the column name from the given csv file name with question IDs.')
parser.add_argument('--output_dir',type=str,default='./model_inference_results',
                    help='Provide the directory for the output files to be saved.')
parser.add_argument('--output_file',type=str,default=None,
                    help='Provide the name of the output file.')
parser.add_argument('--model_cache_dir',type=str,default='.cache',
                    help='Provide the directory saving model caches.')
parser.add_argument("--gpt_azure", type=str2bool, nargs='?',
                    const=True, default=False,
                    help="Whether you are using the AzureOpenAI for GPT-models' response generation.")
parser.add_argument('--temperature',type=float,default=0,
                    help='Provide generation temperature for GPT models.')
parser.add_argument('--top_p',type=int,default=0,
                    help='Provide generation top_p for GPT models.')
parser.add_argument('--gpus',type=str,default=None,
                    help='Provide GPU IDs to use (e.g., "0,1"). Sets CUDA_VISIBLE_DEVICES environment variable.')
parser.add_argument('--num_iterations',type=int,default=1,
                    help='Provide the number of iterations to run.')
parser.add_argument('--sample_size',type=int,default=None,
                    help='Randomly sample N questions per country. If not provided, all questions will be used.')
parser.add_argument('--random_seed',type=int,default=42,
                    help='Random seed for sampling questions. Default is 42 for reproducibility.')
parser.add_argument('--use_persona',type=str2bool,default=True,
                    help='Whether to use persona for response generation. Default is True.')
parser.add_argument('--use_reasoning',type=str2bool,default=True,
                    help='Whether to use reasoning for response generation. Default is True.')

args = parser.parse_args()

def generate_response(model_name,model_path,tokenizer,model,language,country,q_df,q_col,id_col,output_dir,iteration=1, use_persona=True, use_reasoning=True):
    replace_country_flag = False
    if language != COUNTRY_LANG[country] and language == 'English':
        replace_country_flag = True
        
    if q_col == None:
        if language == COUNTRY_LANG[country]:
            q_col = 'Translation'
        elif language == 'English':
            q_col = 'Question'
    
    output_filename = os.path.join(output_dir,f"{model_name}-{country}_{language}_result.csv")
    
    # Load previous iteration results if iteration > 1
    previous_iter_data = {}
    if iteration > 1:
        if os.path.exists(output_filename):
            prev_df = pd.read_csv(output_filename)
            # Filter for previous iteration
            if 'iteration' in prev_df.columns:
                prev_df = prev_df[prev_df['iteration'] == iteration - 1]
                for _, row in prev_df.iterrows():
                    guid = row[id_col]
                    # Store previous persona if available
                    if 'persona' in prev_df.columns:
                        previous_iter_data[guid] = {
                            'persona': row.get('persona', ''),
                            'response': row.get('response', '')
                        }
    
    print(q_df[[id_col,q_col]])
    
    guid_list = set()
    if os.path.exists(output_filename):
        already = pd.read_csv(output_filename)
        # Filter to only current iteration when checking for already processed items
        if 'iteration' in already.columns:
            already = already[already['iteration'] == iteration]
        guid_list = set(already[id_col])
        print(already)
        
        
    else:        
        write_csv_row([id_col,q_col,'prompt','response','iteration','persona','reasoning'],output_filename)
      
    pb = tqdm(q_df.iterrows(),desc=f"{model_name} (iter {iteration})",total=len(q_df))
    for _,d in pb:
        q = d[q_col]
        guid = d[id_col]
        pb.set_postfix({'ID':guid})
        
        if guid in guid_list:
            continue
       
        if replace_country_flag:
            q = replace_country_name(q,country.replace('_',' '))

        prompt = f"Answer the following question.\n\n{q}"
        if use_reasoning:
            prompt += (
                "\n\nRespond in valid JSON format with two keys:\n"
                "\"answer\" (a short answer to the question) and "
                "\"reasoning\" (a short explanation).\n"
                "Example format: {\"answer\": \"{answer here}\", \"reasoning\": \"{reasoning here}\"}\n"
            )
        else:
            prompt += f"Provide one single answer as a list form at the end in a JSON format as below. \n\n{{\"question\":\"{q}\",\"answer\":[\"answer here\"]}} DO NOT include any other text or formatting in your response."

        # Generate or refine persona based on iteration
        persona = None
        if iteration == 1 and use_persona:
            # First iteration: generate new persona
            persona_prompt_formatted = persona_prompt_saq.format(country=country,q=q)
            print("CHAT INPUT INITIAL PERSONA PROMPT (line 129)")
            persona = get_model_response(model_name,persona_prompt_formatted,model,tokenizer,temperature=args.temperature,top_p=args.top_p,gpt_azure=args.gpt_azure)
            print("CHAT OUTPUT INITIAL PERSONA (line 131)")
        elif use_persona:
            # Subsequent iterations: refine previous persona
            prev_data = previous_iter_data.get(guid, {})
            prev_persona = prev_data.get('persona', '')
            prev_response = prev_data.get('response', '')

            # Refine persona (include model's previous answer from previous iteration)
            # Format the system prompt with language and pronoun
            refine_system_prompt = persona_refine_prompt_saq.format(
                language="English",
                second_person_pronoun="You"
            )
            # Create user prompt with question, previous persona, and previous answer
            refine_user_prompt = f"Question: {q}\n\nPrevious persona: {prev_persona}\n\n"
            if prev_response and str(prev_response).strip():
                refine_user_prompt += f"Model's previous answer (from previous iteration): {prev_response}\n\n"
            refine_user_prompt += "Generate the improved persona:"
            print("CHAT INPUT REFINE PERSONA PROMPT (line 149)")
            refine_response = get_model_response(
                model_name,
                refine_user_prompt,
                model,
                tokenizer,
                temperature=args.temperature,
                top_p=args.top_p,
                gpt_azure=args.gpt_azure,
                system_message=refine_system_prompt
            )
            print("CHAT OUTPUT REFINED PERSONA (line 160)")
            # Parse JSON response
            try:
                refine_result = json_repair.loads(refine_response)
                if isinstance(refine_result, dict):
                    persona = refine_result.get("revised_persona", prev_persona)
                elif isinstance(refine_result, str):
                    # Model returned persona directly as string, use it
                    persona = refine_result.strip()
                    if not persona or len(persona) < 10:
                        # If too short, fallback to previous
                        persona = prev_persona
                else:
                    # Unexpected type, fallback
                    persona = prev_persona
            except Exception as e:
                print(f"Error parsing refinement response for {guid}: {e}")
                print(f"Response: {refine_response[:200]}...")  # Print first 200 chars
                # Try to extract persona if it looks like plain text
                refine_response_clean = refine_response.strip()
                if refine_response_clean.startswith("You are") and len(refine_response_clean) > 20:
                    # Looks like a persona, use it
                    persona = refine_response_clean
                else:
                    # Fallback to previous persona if parsing fails
                    persona = prev_persona

        # Use persona as system_message when generating response
        print("CHAT INPUT SEL_OP PROMPT WITH REFINED PERSONA (line 188)")
        response = get_model_response(model_name,prompt,model,tokenizer,temperature=args.temperature,top_p=args.top_p,gpt_azure=args.gpt_azure,system_message=persona)
        print("CHAT OUTPUT NEW ANSWER WITH REFINED PERSONA (line 190)")
        print(response)
        
        # Extract reasoning from JSON response (if available)
        reasoning = ""
        try:
            json_res = get_json_str(response)
            if isinstance(json_res, dict) and 'reasoning' in json_res:
                reasoning = str(json_res['reasoning'])
        except:
            pass  # If parsing fails, leave reasoning empty
        
        write_csv_row([guid,q,prompt,response,iteration,persona,reasoning],output_filename)
        
    del guid_list
            
def get_response_from_all():
    models = args.model
    languages = args.language
    countries = args.country
    question_dir = args.question_dir
    question_col = args.question_col
    id_col = args.id_col
    output_dir = args.output_dir
    use_persona = str2bool(args.use_persona)
    use_reasoning = str2bool(args.use_reasoning)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    if ',' in languages:
        languages = languages.split(',')
        
    if ',' in countries:
        countries = countries.split(',')
        
    if ', ' in models:
        models = models.split(',')
        
    if type(languages) == type(countries) and isinstance(languages,list):
        if len(languages) != len(countries):
            print("ERROR: Same number of languages and countries necessary. If multiple languages and countries are given, each element of the two lists should be in pairs.")
            exit()
        
    def get_questions(language,country):
        questions_df = pd.read_csv(os.path.join(question_dir,f'{country}_questions.csv'),encoding='utf-8')
        
        # Sample questions if sample_size is specified
        if args.sample_size is not None and len(questions_df) > args.sample_size:
            print(f"Sampling {args.sample_size} questions from {len(questions_df)} total questions for {country}")
            questions_df = questions_df.sample(n=args.sample_size, random_state=args.random_seed)
            questions_df = questions_df.reset_index(drop=True)
        
        return questions_df
    
    
    def generate_response_per_model(model_name,use_persona=True,use_reasoning=True):
        # For API-based models (GPT, Claude, Gemini, etc.), use model name as path if not in MODEL_PATHS
        if model_name in MODEL_PATHS:
            model_path = MODEL_PATHS[model_name]
        else:
            # For API models not in MODEL_PATHS, use model name directly
            model_path = model_name
        
        tokenizer,model = get_tokenizer_model(model_name,model_path,args.model_cache_dir)
        
        # Loop over iterations
        for iteration in range(1, args.num_iterations + 1):
            print(f"\n{'='*60}")
            print(f"Starting Iteration {iteration}/{args.num_iterations}")
            print(f"{'='*60}\n")
            
            if isinstance(languages,str):
                questions = get_questions(languages,countries)
                generate_response(model_name,model_path,tokenizer,model,languages,countries,questions,question_col,id_col,output_dir,iteration=iteration,use_persona=use_persona,use_reasoning=use_reasoning)
            else:
                for l,c in zip(languages,countries):
                    questions = get_questions(l,c)
                    generate_response(model_name,model_path,tokenizer,model,l,c,questions,question_col,id_col,output_dir,iteration=iteration,use_persona=use_persona,use_reasoning=use_reasoning)
        
    if isinstance(models,str):
       generate_response_per_model(models,use_persona=use_persona,use_reasoning=use_reasoning)
    else:
        for m in models:
            generate_response_per_model(m,use_persona=use_persona,use_reasoning=use_reasoning)

 
if __name__ == "__main__":
    get_response_from_all()    