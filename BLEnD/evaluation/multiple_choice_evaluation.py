from evaluation_utils import *
from multiple_choice_generation import *
from persona_util import *
import json_repair

def get_model_mc_response(model_name,model_cache_dir,mc_dir,questions_file,response_file=None,temperature=1,top_p=0,gpt_azure=True,num_iterations=1,sample_size=None,random_seed=42):
    if response_file == None:
        response_file = f"{model_name}-mc_res.csv"
    
    questions_df = pd.read_csv(os.path.join(mc_dir,questions_file), encoding='utf-8')
    
    # Sample questions per country if sample_size is specified
    if sample_size is not None and 'country' in questions_df.columns:
        sampled_dfs = []
        for country in questions_df['country'].unique():
            country_df = questions_df[questions_df['country'] == country]
            if len(country_df) > sample_size:
                print(f"Sampling {sample_size} questions from {len(country_df)} total questions for {country}")
                country_sample = country_df.sample(n=sample_size, random_state=random_seed)
            else:
                print(f"Using all {len(country_df)} questions for {country} (less than sample_size)")
                country_sample = country_df
            sampled_dfs.append(country_sample)
        questions_df = pd.concat(sampled_dfs, ignore_index=True)
        print(f"\nTotal questions after sampling: {len(questions_df)}\n")
    
    # For API-based models, use model name as path if not in MODEL_PATHS
    if model_name in MODEL_PATHS:
        model_path = MODEL_PATHS[model_name]
    else:
        model_path = model_name
    
    tokenizer,model = get_tokenizer_model(model_name,model_path,model_cache_dir)
    
    # Loop through iterations
    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*60}")
        print(f"Starting Iteration {iteration}/{num_iterations}")
        print(f"{'='*60}\n")
        
        # Load previous iteration results if iteration > 1
        previous_iter_data = {}
        if iteration > 1:
            if os.path.exists(os.path.join(mc_dir,response_file)):
                prev_df = pd.read_csv(os.path.join(mc_dir,response_file),encoding='utf-8')
                # Filter for previous iteration
                if 'iteration' in prev_df.columns:
                    prev_df = prev_df[prev_df['iteration'] == iteration - 1]
                    for _, row in prev_df.iterrows():
                        qid = row['MCQID']
                        # Store previous persona if available
                        if 'persona' in prev_df.columns:
                            previous_iter_data[qid] = {
                                'persona': row.get('persona', ''),
                                'response': row.get('full_res', '')
                            }
        
        # Check what's already done for current iteration
        already = None
        if not os.path.exists(os.path.join(mc_dir,response_file)):
            write_csv_row(list(questions_df.columns)+['full_res','final_ans','iteration','persona'],os.path.join(mc_dir,response_file))
        else:
            already = pd.read_csv(os.path.join(mc_dir,response_file),encoding='utf-8')
            # Filter to only current iteration when checking for already processed items
            if 'iteration' in already.columns:
                already = already[already['iteration'] == iteration]

        pb = tqdm(questions_df.iterrows(),total=len(questions_df),desc=f"{model_name} (iter {iteration})")
        right = 0
        for i,row in pb:
            
            qid = row['MCQID']
            pb.set_postfix({'ID':qid})
            
            if isinstance(already,pd.DataFrame):
                if qid in set(already['MCQID']):
                    continue
            
            country = row['country']
            prompt = row['prompt']
            print(prompt)

            # Generate or refine persona based on iteration
            if iteration == 1:
                # First iteration: generate new persona
                persona_prompt_formatted = generate_persona_prompt + f"\n\nCountry: {country}\nQuestion: {prompt}\n\nGenerate the persona:"
                persona = get_model_response(model_name,persona_prompt_formatted,model,tokenizer,temperature,top_p,gpt_azure)
            else:
                # Subsequent iterations: refine previous persona
                prev_persona = previous_iter_data.get(qid, {}).get('persona', '')
                if not prev_persona:
                    # Fallback: if no previous persona found, generate new one
                    print(f"Warning: No previous persona found for {qid}, generating new persona")
                    persona_prompt_formatted = generate_persona_prompt + f"\n\nCountry: {country}\nQuestion: {prompt}\n\nGenerate the persona:"
                    persona = get_model_response(model_name,persona_prompt_formatted,model,tokenizer,temperature,top_p,gpt_azure)
                else:
                    # Refine persona using MC-specific prompt
                    # Format the system prompt with language and pronoun
                    refine_system_prompt = persona_refine_prompt_mcq.format(
                        language="English",
                        second_person_pronoun="You"
                    )
                    # Create user prompt with question and previous persona
                    refine_user_prompt = f"Question: {prompt}\n\nPrevious persona: {prev_persona}\n\nGenerate the improved persona:"
                    refine_response = get_model_response(
                        model_name,
                        refine_user_prompt,
                        model,
                        tokenizer,
                        temperature,
                        top_p,
                        gpt_azure,
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
                        print(f"Error parsing refinement response for {qid}: {e}")
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

            full_res = get_model_response(model_name,prompt,model,tokenizer,temperature,top_p,gpt_azure,system_message=persona)
            print(full_res)
            json_res = get_json_str(full_res)
            
            if isinstance(json_res,dict) and 'answer_choice' in json_res:
                try:
                    final_ans = re.findall(r'[A-Z]',str(json_res['answer_choice']))[0]
                    if final_ans+'.' not in prompt:
                        for k,v in json.loads(row['choices']).items():
                            if v == json_res['answer_choice']:
                                final_ans = str(k)
                                break
                        else:
                            final_ans = full_res 
                    
                except:
                    for k,v in json.loads(row['choices']).items():
                        if v == json_res['answer_choice']:
                            final_ans = str(k)
                            break
                    else:
                        final_ans = full_res
            else:
                try:
                    final_ans = re.findall(r'[A-Z]',json_res)[0]
                except:
                    final_ans = full_res
            
            write_csv_row(list(row)+[full_res,final_ans,iteration,persona],os.path.join(mc_dir,response_file))
            if final_ans == row['answer_idx']:
                right += 1
            pb.set_postfix({'score':right/(i+1)})

def multiple_choice_score(model,mc_dir,mrf,mc_res_file,eval_res_file,wrong_country_ratio_file,country):
    
    df = pd.read_csv(os.path.join(mc_dir,mrf),encoding='utf-8')
    df = df[df['country'] == country]
    
    scores = []
    
    for i,row in tqdm(df.iterrows(),total=len(df)):
        if str(row['answer_idx']) == str(row['final_ans']):
            scores.append(1)
        else:
            scores.append(0)
            
        
    df['score'] = scores
    final_score = df['score'].mean()
    
    return final_score
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose your model(s) & language(s)')
    
    parser.add_argument('--model',type=str,
                        help='Provide the model you want to use. Check and choose from the key values of the MODEL_PATHS variable. If you want to test on multiple models, provide multiple model names with ", " between each (e.g., "gpt-4-0125-preview, aya-101").')
    parser.add_argument('--model_cache_dir',type=str,default='.cache',
                    help='Provide the directory saving model caches.')
    
    parser.add_argument('--mc_dir',type=str,default='./mc_data',
                        help='Provide the directory for the data files from the human annotators.')
    parser.add_argument('--questions_file',type=str,default='mc_questions_file.csv',
                        help='Provide the directory for the data files from the human annotators.')
    parser.add_argument('--response_file',type=str,default=None,
                        help='Provide the filename to save LLM responses.')
    
    parser.add_argument('--temperature',type=float,default=0,
                    help='Provide generation temperature for LLMs.')
    parser.add_argument('--top_p',type=float,default=1,
                    help='Provide generation top_p for LLMs.')
    
    parser.add_argument("--gpt_azure", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Whether you are using the AzureOpenAI for GPT-models' response generation.")
    parser.add_argument('--num_iterations',type=int,default=1,
                        help='Provide the number of iterations to run.')
    parser.add_argument('--sample_size',type=int,default=None,
                        help='Number of questions to sample per country. If None, use all questions.')
    parser.add_argument('--random_seed',type=int,default=42,
                        help='Random seed for sampling questions.')
    
    args = parser.parse_args()
    
    get_model_mc_response(model_name=args.model,
                          model_cache_dir=args.model_cache_dir,
                          mc_dir=args.mc_dir,
                          questions_file=args.questions_file,
                          response_file=args.response_file,
                          temperature=args.temperature,
                          top_p=args.top_p,
                          gpt_azure=args.gpt_azure,
                          num_iterations=args.num_iterations,
                          sample_size=args.sample_size,
                          random_seed=args.random_seed)