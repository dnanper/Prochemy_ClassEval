import shutil
import time
from func_timeout import func_set_timeout
import func_timeout
import importlib
import unittest
import json
import re
import os
from scipy.special import comb
from path_util import PathUtil
from test_pipeline import AutoTest

class CustomAutoTest(AutoTest):
    
    def __init__(self, eval_data_name):
        super().__init__(eval_data_name)

    def gen_code_list(self, file_path):
        code_list = []

        # JSONL file
        with open(file_path, 'r', encoding="utf-8") as f:
            for line in f:
                code_list.append(json.loads(line))
        return code_list

    def gen_py_file(self, test_code_name, code, method_name, test_code):
        test_name = test_code_name + '_' + method_name + '.py'
        test_code_py = code + '\n' + test_code
        with open(test_name, 'w', encoding='utf-8') as f:
            f.write(test_code_py)
        
    def test(self, test_code_name, method_name, model_name):
        result = {}
        test_module_name = test_code_name + '_' + method_name
        task_id = test_code_name
        test_class = None

        for method in self.eval_data[task_id]['methods_info']:
            if method['method_name'] == method_name:
                test_class = method['test_class']
                break
        
        if not test_class:
            return result

        try:
            res = self.run_unit_test(test_module_name, test_class, model_name)
            result[test_class] = {
                'errors': len(res.errors),
                'failures': len(res.failures),
                'testsRun': res.testsRun
            }
        except func_timeout.exceptions.FunctionTimedOut:
            print(f"⏱️  TIMEOUT (30s) for {test_module_name}.{test_class}")
            result[test_class] = {
                'errors': 0,
                'failures': 0,
                'testsRun': 0
            }
        except Exception as e:
            print(f"❌ ERROR in test() for {test_module_name}.{test_class}: {e}")
            import traceback
            traceback.print_exc()
            result[test_class] = {
                'errors': 0,
                'failures': 0,
                'testsRun': 0
            }

        return result
        
    def test_pipeline(self, model_name, gen_file_path):
        result_dict = {}

        task_list = self.gen_code_list(gen_file_path)

        # Gen python file
        for task in task_list:
            task_id = task['task_id']
            method_name = task['method_name']
            test_code = "import unittest"
            # TODO them test code vao generation
            for method in self.eval_data[task_id]['methods_info']:
                if method['method_name'] == method_name:
                    test_code += '\n\n' + method['test_code']
            
            self.gen_py_file(task_id, task['class_code'], method_name, test_code)

        # Run unit test
        for task in task_list:
            task_id = task['task_id']
            method_name = task['method_name']            
            try:
                result = self.test(task_id, method_name, model_name)
                result_key = task_id + '_' + method_name
                result_dict[result_key] = result
            except Exception as e:
                print(f"ERROR for {task_id}_{method_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        self.save_result(model_name, result_dict, "method")
        time.sleep(5)
        self.tear_down()

    def evaluate(self, model_list):
        """
        Evaluate method-level test results for multiple models.
        Structure: {model_name: {task_id_methodname: {test_class: {errors, failures, testsRun}}}}
        """
        result_dict = {}
        
        for model_name in model_list:
            model_result_path = PathUtil().test_result_data(
                model_name + '_method_result', 'json')
            with open(model_result_path, 'r') as f:
                model_result = json.load(f)
            
            result_dict[model_name] = {}
            
            # Loop through each task_id_methodname
            for task_method_key in model_result:
                result_dict[model_name][task_method_key] = {}
                
                # Each task_method_key has one test_class
                for test_class in model_result[task_method_key]:
                    test_result = model_result[task_method_key][test_class]
                    test_answer = self.get_test_answer(test_result)
                    
                    result_dict[model_name][task_method_key][test_class] = {
                        'result': test_answer,
                        'errors': test_result['errors'],
                        'failures': test_result['failures'],
                        'testsRun': test_result['testsRun']
                    }
        
        save_path = PathUtil().test_result_data("detailed_method_result", 'json')
        with open(save_path, 'w') as f:
            json.dump(result_dict, f, indent=4, sort_keys=True)
        
        return result_dict
    
    def cal_metrics(self, model_list):
        """
        Calculate success/fail metrics for method-level testing.
        Returns: {model_name: {success_rate, partial_success_rate, fail_rate, error_rate, total_methods}}
        """
        file_path = PathUtil().test_result_data("detailed_method_result", 'json')
        with open(file_path, 'r') as f:
            detailed_result = json.load(f)
        
        metrics = {}
        
        for model_name in model_list:
            success_count = 0
            partial_success_count = 0
            fail_count = 0
            error_count = 0
            total_count = 0
            
            for task_method_key in detailed_result[model_name]:
                for test_class in detailed_result[model_name][task_method_key]:
                    result = detailed_result[model_name][task_method_key][test_class]['result']
                    total_count += 1
                    
                    if result == 'success':
                        success_count += 1
                    elif result == 'partial_success':
                        partial_success_count += 1
                    elif result == 'fail':
                        fail_count += 1
                    elif result == 'error':
                        error_count += 1
            
            metrics[model_name] = {
                'success_rate': success_count / total_count if total_count > 0 else 0,
                'partial_success_rate': partial_success_count / total_count if total_count > 0 else 0,
                'fail_rate': fail_count / total_count if total_count > 0 else 0,
                'error_rate': error_count / total_count if total_count > 0 else 0,
                'total_methods': total_count,
                'success_count': success_count,
                'partial_success_count': partial_success_count,
                'fail_count': fail_count,
                'error_count': error_count
            }
        
        return metrics

        
