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
parser.add_argument('--prompt_dir',type=str,default=None,
                    help='Provide the directory where the propmts are saved.')
parser.add_argument('--prompt_file',type=str,default=None,
                    help='Provide the name of the csv file where the propmts are saved.')
parser.add_argument('--prompt_no',type=str,default=None,
                    help='Provide the propmt id (ex. inst-1, inst-2, pers-1, etc.)')
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

args = parser.parse_args()

def make_prompt(question,prompt_no,language,country,prompt_sheet):
    prompt = prompt_sheet[prompt_sheet['id']==prompt_no]
    if language == 'English':
        prompt = prompt['English'].values[0]
    else:
        prompt = prompt['Translation'].values[0]

    return prompt.replace('{q}',question)

def generate_response(model_name,model_path,tokenizer,model,language,country,q_df,q_col,id_col,output_dir,prompt_no=None,prompt_dir=None,prompt_file=None,iteration=1):
    replace_country_flag = False
    if language != COUNTRY_LANG[country] and language == 'English':
        replace_country_flag = True
        
    if q_col == None:
        if language == COUNTRY_LANG[country]:
            q_col = 'Translation'
        elif language == 'English':
            q_col = 'Question'
    
    if prompt_no is not None:
        # Load prompts from local CSV file or Google Sheets
        if prompt_dir and prompt_file:
            # Use specified prompt file
            prompt_filepath = os.path.join(prompt_dir, prompt_file)
            if os.path.exists(prompt_filepath):
                prompt_sheet = pd.read_csv(prompt_filepath, encoding='utf-8')
            else:
                raise FileNotFoundError(f"Prompt file not found: {prompt_filepath}")
        elif prompt_dir:
            # Use default prompt filename pattern: {country}_prompts.csv
            prompt_filepath = os.path.join(prompt_dir, f'{country}_prompts.csv')
            if os.path.exists(prompt_filepath):
                prompt_sheet = pd.read_csv(prompt_filepath, encoding='utf-8')
            else:
                raise FileNotFoundError(f"Prompt file not found: {prompt_filepath}. Please provide --prompt_dir or set PROMPT_SHEET_ID and PROMPT_COUNTRY_SHEET for Google Sheets.")
        else:
            # Try to use Google Sheets (will fail if not configured)
            try:
                prompt_sheet = import_google_sheet(PROMPT_SHEET_ID, PROMPT_COUNTRY_SHEET[country])
            except NameError:
                raise NameError("Either provide --prompt_dir with prompt CSV files, or set PROMPT_SHEET_ID and PROMPT_COUNTRY_SHEET for Google Sheets.")
        output_filename = os.path.join(output_dir,f"{model_name}-{country}_{language}_{prompt_no}_result.csv")
    else:
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
        write_csv_row([id_col,q_col,'prompt','response','prompt_no','iteration','persona'],output_filename)
      
    pb = tqdm(q_df.iterrows(),desc=f"{model_name} (iter {iteration})",total=len(q_df))
    for _,d in pb:
        q = d[q_col]
        guid = d[id_col]
        pb.set_postfix({'ID':guid})
        
        if guid in guid_list:
            continue
       
        if replace_country_flag:
            q = replace_country_name(q,country.replace('_',' '))
       
        if prompt_no is not None:
            prompt = make_prompt(q,prompt_no,language,country,prompt_sheet)
        else:
            prompt = q
        
        # Generate or refine persona based on iteration
        if iteration == 1:
            # First iteration: generate new persona
            persona_prompt_formatted = persona_prompt_saq.format(country=country,q=q)
            persona = get_model_response(model_name,persona_prompt_formatted,model,tokenizer,temperature=args.temperature,top_p=args.top_p,gpt_azure=args.gpt_azure)
        else:
            # Subsequent iterations: refine previous persona
            prev_persona = previous_iter_data.get(guid, {}).get('persona', '')
            if not prev_persona:
                # Fallback: if no previous persona found, generate new one
                print(f"Warning: No previous persona found for {guid}, generating new persona")
                persona_prompt_formatted = persona_prompt_saq.format(country=country,q=q)
                persona = get_model_response(model_name,persona_prompt_formatted,model,tokenizer,temperature=args.temperature,top_p=args.top_p,gpt_azure=args.gpt_azure)
            else:
                # Refine persona
                # Format the system prompt with language and pronoun
                refine_system_prompt = persona_refine_prompt_saq.format(
                    language="English",
                    second_person_pronoun="You"
                )
                # Create user prompt with question and previous persona
                refine_user_prompt = f"Question: {q}\n\nPrevious persona: {prev_persona}\n\nGenerate the improved persona:"
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

        print("--------------------------------")
        print("Persona: ",persona)
        print("--------------------------------")
        print(prompt)
        
        # Use persona as system_message when generating response
        response = get_model_response(model_name,prompt,model,tokenizer,temperature=args.temperature,top_p=args.top_p,gpt_azure=args.gpt_azure,system_message=persona)
            
        print(response)
        write_csv_row([guid,q,prompt,response,prompt_no,iteration,persona],output_filename)
        
    del guid_list
            
def get_response_from_all():
    models = args.model
    languages = args.language
    countries = args.country
    question_dir = args.question_dir
    question_file = args.question_file
    question_col = args.question_col
    prompt_no = args.prompt_no
    prompt_dir = args.prompt_dir
    prompt_file = args.prompt_file
    id_col = args.id_col
    output_dir = args.output_dir
    azure = args.gpt_azure
    
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
    
    
    def generate_response_per_model(model_name):
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
                generate_response(model_name,model_path,tokenizer,model,languages,countries,questions,question_col,id_col,output_dir,prompt_no=prompt_no,prompt_dir=prompt_dir,prompt_file=prompt_file,iteration=iteration)
            else:
                for l,c in zip(languages,countries):
                    questions = get_questions(l,c)
                    generate_response(model_name,model_path,tokenizer,model,l,c,questions,question_col,id_col,output_dir,prompt_no=prompt_no,prompt_dir=prompt_dir,prompt_file=prompt_file,iteration=iteration)
        
    if isinstance(models,str):
       generate_response_per_model(models)
    else:
        for m in models:
            generate_response_per_model(m)

 
if __name__ == "__main__":
    get_response_from_all()    