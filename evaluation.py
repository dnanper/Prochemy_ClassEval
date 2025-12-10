import argparse
import json
import os
from test_pipeline import AutoTest
from custom_test_pipeline import CustomAutoTest
from path_util import PathUtil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_file_name",
        type=str,
        default="Qwen7B_Prochemy_infile",
        help="source of model output",
    )
    parser.add_argument(
        "--greedy",
        type=int,
        default=1,
        help="whether the model result is greedy or not",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default='ClassEval_data',
        help="ClassEval data",
    )

    parser.add_argument(
        "--custom_eval",
        type=bool,
        default=True,
        help="CustomEval for ZeroShot + Infile-contextt",
    )

    args = parser.parse_args()

    model_list = [args.source_file_name]

    if args.custom_eval:
        # Method-level evaluation (ZeroShot + InFile context)
        CustomAutoT = CustomAutoTest(args.eval_data)
        
        for model_name in model_list:
            file_path = PathUtil().model_output_data(model_name, "jsonl")
            CustomAutoT.test_pipeline(model_name, file_path)
        
        CustomAutoT.evaluate(model_list)
        metrics = CustomAutoT.cal_metrics(model_list)
        
        # Save custom eval metrics
        save_path = PathUtil().test_result_data("method_level_metrics", 'json')
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4, sort_keys=True)
        
        print("\n=== Method-Level Evaluation Results ===")
        for model_name in model_list:
            m = metrics[model_name]
            print(f"\nModel: {model_name}")
            print(f"  Total Methods: {m['total_methods']}")
            print(f"  Success: {m['success_count']} ({m['success_rate']:.2%})")
            print(f"  Partial Success: {m['partial_success_count']} ({m['partial_success_rate']:.2%})")
            print(f"  Fail: {m['fail_count']} ({m['fail_rate']:.2%})")
            print(f"  Error: {m['error_count']} ({m['error_rate']:.2%})")
    
    else:
        # Class-level evaluation (original multi-prediction setup)
        AutoT = AutoTest(args.eval_data)
        
        for model_name in model_list:
            file_path = PathUtil().model_output_data(model_name, "json")
            AutoT.test_pipeline(model_name, file_path)

        AutoT.evaluate(model_list)
        result = {}
        
        if args.greedy == 1:
            result["pass_1_greedy"] = AutoT.cal_metrics_pass_at_k(model_list, 1, 1)
        else:
            result["pass_1"] = AutoT.cal_metrics_pass_at_k(model_list, 1, 5)
            result["pass_3"] = AutoT.cal_metrics_pass_at_k(model_list, 3, 5)
            result["pass_5"] = AutoT.cal_metrics_pass_at_k(model_list, 5, 5)
        
        save_path = PathUtil().test_result_data("pass_at_k_result", 'json')

        if os.path.exists(save_path):
            with open(save_path, encoding='utf-8') as file:
                ori_data = json.load(file)

            if args.greedy == 1:
                if "pass_1_greedy" in ori_data:
                    ori_data["pass_1_greedy"][args.source_file_name] = result["pass_1_greedy"][args.source_file_name]
                else:
                    ori_data["pass_1_greedy"] = result["pass_1_greedy"]
            else:
                if "pass_1" in ori_data:
                    ori_data["pass_1"][args.source_file_name] = result["pass_1"][args.source_file_name]
                    ori_data["pass_3"][args.source_file_name] = result["pass_3"][args.source_file_name]
                    ori_data["pass_5"][args.source_file_name] = result["pass_5"][args.source_file_name]
                else:
                    ori_data["pass_1"] = result["pass_1"]
                    ori_data["pass_3"] = result["pass_3"]
                    ori_data["pass_5"] = result["pass_5"]
        else:
            ori_data = result

        with open(save_path, 'w') as f:
            json.dump(ori_data, f, indent=4, sort_keys=True)
        
        print("\n=== Class-Level Pass@k Results ===")
        print(json.dumps(result, indent=2))