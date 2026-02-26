import re
from evaluation_utils import *
from exact_match import *
from multiple_choice_evaluation import *
from saq_llm_judge import saq_llm_judge

ALL_QUESTIONS_CSV = "saq_llm_judge_all_questions.csv"
ACCURACY_BY_ITER_CSV = "saq_llm_judge_accuracy_by_iteration.csv"


def evaluate_all_metrics(
    model,country,language,
    response_dir,annotation_dir,mc_dir,
    id_col,q_col,r_col,annotations_key,
    eval_res_filename,annotation_template='{country}_data.json'
    ):
    # Write SAQ results under saq_results/<model>/
    saq_out_dir = os.path.join('saq_results', model)
    os.makedirs(saq_out_dir, exist_ok=True)
    # Derive run suffix from eval filename (e.g. baseline_r1 from llama3-8b-instruct_baseline_r1_results.csv)
    base = os.path.splitext(os.path.basename(eval_res_filename))[0]
    run_suffix = ""
    if "baseline_r" in base:
        m = re.search(r"baseline_r\d+", base, re.I)
        if m:
            run_suffix = "_" + m.group(0)
    all_questions_path = os.path.join(saq_out_dir, ALL_QUESTIONS_CSV.replace(".csv", f"{run_suffix}.csv"))
    accuracy_by_iter_path = os.path.join(saq_out_dir, ACCURACY_BY_ITER_CSV.replace(".csv", f"{run_suffix}.csv"))

    res_df = get_model_response_file(data_dir=response_dir,model=model,country=country,language=language)
    
    # Check if iteration column exists and has valid (non-NaN) values
    has_iteration = 'iteration' in res_df.columns
    if has_iteration:
        iteration_vals = res_df['iteration'].dropna().unique()
        if len(iteration_vals) > 0:
            iterations = sorted(iteration_vals)
            print(f"Found iterations: {iterations}")
        else:
            has_iteration = False
            iterations = [None]
    if not has_iteration:
        iterations = [None]
    
    real_annotation = get_annotations(data_dir=annotation_dir,country=country,template=annotation_template)
    
    iteration_scores = {}
    all_question_rows = []
    
    for iteration in iterations:
        if has_iteration and iteration is not None:
            iter_df = res_df[res_df['iteration'] == iteration].copy()
            print(f"\n{'='*60}")
            print(f"Evaluating Iteration {iteration}")
            print(f"{'='*60}\n")
        else:
            iter_df = res_df.copy()
            print(f"\n{'='*60}")
            print(f"Evaluating All Data (no iteration column)")
            print(f"{'='*60}\n")
        
        if len(iter_df) == 0:
            print(f"Warning: No data found for iteration {iteration}. Skipping.")
            continue

        accuracy, num_correct, num_total, question_rows = saq_llm_judge(
            country=country,
            language=language,
            annotation_dict=real_annotation,
            response_df=iter_df,
            id_col=id_col,
            r_col=r_col,
            annotations_key=annotations_key,
            judge_model=model,
        )
        
        print(f"  Accuracy: {accuracy:.2f}% ({num_correct}/{num_total})")

        iteration_key = f"iteration_{iteration}" if iteration is not None else "all"
        iteration_scores[iteration_key] = {'accuracy': accuracy, 'num_correct': num_correct, 'num_total': num_total}
        
        for row in question_rows:
            row["model"] = model
            all_question_rows.append(row)
    
    # Append to big fat CSV (create with headers if new)
    if all_question_rows:
        headers = ["model", "country", "iteration", "question_id", "correct", "model_response", "ground_truth"]
        file_exists = os.path.exists(all_questions_path)
        with open(all_questions_path, "a", encoding="utf-8") as f:
            import csv
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            for row in all_question_rows:
                writer.writerow([
                    row.get("model", model),
                    row.get("country", country),
                    row.get("iteration", ""),
                    row.get("question_id", ""),
                    row.get("correct", 0),
                    (row.get("model_response", "") or "")[:500],
                    (row.get("ground_truth", "") or "")[:500],
                ])
        
        # Recompute and overwrite accuracy-by-iteration summary (aggregate across all countries so far)
        try:
            df_all = pd.read_csv(all_questions_path, encoding="utf-8")
            summary_rows = []
            if "iteration" in df_all.columns:
                it_vals = [v for v in df_all["iteration"].dropna().unique() if v != "" and str(v).strip() != ""]
                def _sort_key(x):
                    try:
                        return float(x)
                    except (TypeError, ValueError):
                        return 0
                it_vals = sorted(it_vals, key=_sort_key)
                for it in it_vals:
                    sub = df_all[df_all["iteration"].astype(str) == str(it)]
                    if len(sub) > 0:
                        acc = sub["correct"].mean() * 100
                        n_corr = int(sub["correct"].sum())
                        n_tot = len(sub)
                        summary_rows.append({"iteration": it, "accuracy": acc, "num_correct": n_corr, "num_total": n_tot})
            if summary_rows:
                pd.DataFrame(summary_rows).to_csv(accuracy_by_iter_path, index=False, encoding="utf-8")
        except Exception as e:
            print(f"Warning: could not update accuracy-by-iteration: {e}")
    
    print(f"\n{'='*60}")
    print(f"Evaluation Summary for {model} - {country} - {language}")
    print(f"{'='*60}")
    for iter_key, scores in sorted(iteration_scores.items()):
        print(f"{iter_key.upper()}: Accuracy {scores['accuracy']:.2f}% ({scores['num_correct']}/{scores['num_total']})")
    print(f"{'='*60}\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose your model(s) & language(s)')
    parser.add_argument('--model',type=str,
                        help='Provide the model you want to use. Check and choose from the key values of the MODEL_PATHS variable. If you want to test on multiple models, provide multiple model names with ", " between each (e.g., "gpt-4-0125-preview, aya-101").')
    parser.add_argument('--language',type=str,default=None,
                        help='Provide the language you want to test on. Check and choose from the first values of the LANG_COUNTRY variable. If you want to test on multiple languages, provide multiple languages with ", " between each (e.g., "English, Korean").')
    parser.add_argument('--country',type=str,default=None,
                        help='Provide the country you want to test on. Check and choose from the second values of the LANG_COUNTRY variable. If you want to test on multiple countries, provide multiple countries with ", " between each (e.g., "UK, South Korea"). Make sure you have the same number of countries and languages provided. The language-country pair do not have to be identical with the pairs within the LANG_COUNTRY variable.')
    
    parser.add_argument('--id_col',type=str,default=None,
                        help='Provide the column name from the LLM response csv file name with question IDs.') 
    parser.add_argument('--question_col',type=str,default=None,
                        help='Provide the column name from the LLM response csv file name with questions.')
    parser.add_argument('--response_col',type=str,default=None,
                        help='Provide the column name from the LLM response csv file name with LLM responses.') 

    parser.add_argument('--response_dir',type=str,default='../model_inference_results',
                        help='Provide the directory for the output files to be saved.')
    parser.add_argument('--annotation_dir',type=str,default='../final_dataset',
                        help='Provide the directory for the data files from the human annotators.')
    parser.add_argument('--mc_dir',type=str,default='./mc_data',
                        help='Provide the directory for the multiple choice result files.')
    parser.add_argument('--annotation_filename',type=str,default='{country}_data.json',)
    parser.add_argument('--annotations_key',type=str,default='annotations',
                        help='Provide the key for the annotations in the annotation file.')
    parser.add_argument('--evaluation_result_file',type=str,default='evaluation_results.csv',
                        help='Provide the filename for the evaluation result file.')
    
    args = parser.parse_args()
    
    evaluate_all_metrics(model=args.model,country=args.country,language=args.language,response_dir=args.response_dir,annotation_dir=args.annotation_dir,mc_dir=args.mc_dir,id_col=args.id_col,q_col=args.question_col,r_col=args.response_col,eval_res_filename=args.evaluation_result_file,annotations_key=args.annotations_key,annotation_template=args.annotation_filename) 