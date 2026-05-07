import json
import os
from openai import AzureOpenAI
import time
from typing import List, Dict, Any
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

class MoleculeInferenceProcessor:
    def __init__(self, endpoint=None, api_key=None, model_name="gpt-4o", deployment="gpt-4o", api_version="2024-12-01-preview", max_workers=5):
        """
        Initialize the Molecule Inference Processor
        
        Args:
            endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            model_name: Model name to use
            deployment: Deployment name
            api_version: API version
            max_workers: Maximum number of parallel workers
        """
        # Use provided values or defaults from your reference code
        self.endpoint = endpoint or "https://prometheus15.openai.azure.com/"
        self.model_name = model_name
        self.deployment = deployment
        self.subscription_key = api_key or os.environ.get("AZURE_OPENAI_KEY", "")
        self.api_version = api_version
        self.max_workers = max_workers
        
        # Thread-local storage for OpenAI clients
        self._local = threading.local()
        
        # Lock for thread-safe file writing
        self._file_lock = threading.Lock()
    
    def get_client(self):
        """Get thread-local OpenAI client"""
        if not hasattr(self._local, 'client'):
            self._local.client = AzureOpenAI(
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                api_key=self.subscription_key,
            )
        return self._local.client
    
    def create_prompt(self, instruction: str, input_smiles: str) -> str:
        """
        Create a prompt by combining instruction and input
        
        Args:
            instruction: The instruction text
            input_smiles: The SMILES string input
            
        Returns:
            Combined prompt string
        """
        # Format similar to your example
        prompt = f"{instruction}\n{input_smiles}"
        return prompt
    
    def get_model_response(self, instruction: str, input_smiles: str, max_retries: int = 3) -> str:
        """
        Get model response for a given instruction and input
        
        Args:
            instruction: The instruction text
            input_smiles: The SMILES string
            max_retries: Maximum number of retry attempts
            
        Returns:
            Model response string
        """
        client = self.get_client()
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are GPT-4o, created by OpenAI. You are a helpful assistant."
                        },
                        {
                            "role": "user", 
                            "content": f"{instruction}\n{input_smiles}"
                        }
                    ],
                    max_tokens=1500,
                    temperature=0.1,  # Low temperature for consistent results
                    top_p=0.9,
                    model=self.deployment
                )
                
                return response.choices[0].message.content.strip()
            
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return f"Error: API call failed after {max_retries} attempts: {str(e)}"
    
    def load_test_data(self, input_file: str) -> List[Dict[str, Any]]:
        """
        Load test data from JSON file
        
        Args:
            input_file: Path to input JSON file
            
        Returns:
            List of test data items
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Successfully loaded {len(data)} items from {input_file}")
            return data
        except FileNotFoundError:
            print(f"Error: File '{input_file}' not found.")
            return []
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in file '{input_file}': {str(e)}")
            return []
        except Exception as e:
            print(f"Error loading test data: {str(e)}")
            return []
    
    def process_single_item(self, item_data: tuple) -> Dict[str, Any]:
        """
        Process a single item (used for parallel processing)
        
        Args:
            item_data: Tuple of (index, item_dict)
            
        Returns:
            Result dictionary
        """
        index, item = item_data
        
        # Extract instruction, input, and expected output
        instruction = item.get('instruction', '')
        input_smiles = item.get('input', '')
        expected_output = item.get('output', '')
        
        # Create prompt
        prompt = self.create_prompt(instruction, input_smiles)
        
        # Get model prediction
        prediction = self.get_model_response(instruction, input_smiles)
        
        # Create result entry
        result = {
            "index": index,
            "prompt": prompt,
            "predict": prediction,
            "label": expected_output
        }
        
        return result
    
    def process_inference_parallel(self, data_list: List[Dict[str, Any]], output_file: str):
        """
        Process inference with parallel processing and save results
        
        Args:
            data_list: List of test data items
            output_file: Output file path
        """
        print(f"开始并行处理 {len(data_list)} 个项目，使用 {self.max_workers} 个工作线程...")
        print(f"Starting parallel processing of {len(data_list)} items with {self.max_workers} workers...")
        
        results = [None] * len(data_list)  # Pre-allocate list to maintain order
        successful_count = 0
        
        # Prepare data with indices
        indexed_data = [(i, item) for i, item in enumerate(data_list)]
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.process_single_item, item_data): item_data[0] 
                for item_data in indexed_data
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(data_list), desc="Processing items") as pbar:
                for future in as_completed(future_to_index):
                    try:
                        result = future.result()
                        index = result.pop('index')  # Remove index from final result
                        results[index] = result
                        successful_count += 1
                        pbar.update(1)
                    except Exception as e:
                        index = future_to_index[future]
                        print(f"Error processing item {index}: {str(e)}")
                        pbar.update(1)
        
        # Filter out None results and save
        valid_results = [r for r in results if r is not None]
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(valid_results, f, ensure_ascii=False, indent=2)
            print(f"\n🎉 Successfully processed and saved {successful_count}/{len(data_list)} items to {output_file}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
        
        return successful_count
    
    def process_inference_parallel_streaming(self, data_list: List[Dict[str, Any]], output_file: str):
        """
        Process inference with parallel processing and save each result immediately to JSONL
        
        Args:
            data_list: List of test data items
            output_file: JSONL output file path
        """
        print(f"开始并行流式处理 {len(data_list)} 个项目，使用 {self.max_workers} 个工作线程...")
        print(f"Starting parallel streaming processing of {len(data_list)} items with {self.max_workers} workers...")
        
        # Clear the output file first
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                pass  # Just create/clear the file
        except Exception as e:
            print(f"Error creating output file: {str(e)}")
            return
        
        successful_count = 0
        
        # Prepare data with indices
        indexed_data = [(i, item) for i, item in enumerate(data_list)]
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.process_single_item, item_data): item_data[0] 
                for item_data in indexed_data
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(data_list), desc="Processing items") as pbar:
                for future in as_completed(future_to_index):
                    try:
                        result = future.result()
                        index = result.pop('index')  # Remove index from final result
                        
                        # Thread-safe file writing
                        with self._file_lock:
                            with open(output_file, 'a', encoding='utf-8') as f:
                                json.dump(result, f, ensure_ascii=False)
                                f.write('\n')
                        
                        successful_count += 1
                        pbar.set_postfix({'completed': successful_count})
                        pbar.update(1)
                        
                    except Exception as e:
                        index = future_to_index[future]
                        print(f"Error processing item {index}: {str(e)}")
                        pbar.update(1)
        
        print(f"\n🎉 Successfully processed and saved {successful_count}/{len(data_list)} items to {output_file}")
        return successful_count
    
    def process_inference_batch(self, data_list: List[Dict[str, Any]], output_file: str):
        """
        Process inference for all test data and save results (Sequential version)
        
        Args:
            data_list: List of test data items
            output_file: Output file path
        """
        results = []
        successful_count = 0
        
        for i, item in enumerate(tqdm(data_list, desc="Processing items")):
            print(f"Processing item {i+1}/{len(data_list)}...")
            
            # Extract instruction, input, and expected output
            instruction = item.get('instruction', '')
            input_smiles = item.get('input', '')
            expected_output = item.get('output', '')
            
            # Create prompt
            prompt = self.create_prompt(instruction, input_smiles)
            
            # Get model prediction
            prediction = self.get_model_response(instruction, input_smiles)
            
            # Create result entry
            result = {
                "prompt": prompt,
                "predict": prediction,
                "label": expected_output
            }
            
            results.append(result)
            successful_count += 1
            
            print(f"✓ Processed item {i+1}")
            
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Save all results to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n🎉 Successfully processed and saved {successful_count}/{len(data_list)} items to {output_file}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
        
        return successful_count
    
    def process_inference_streaming(self, data_list: List[Dict[str, Any]], output_file: str):
        """
        Process inference and save each result immediately to JSONL format
        
        Args:
            data_list: List of test data items
            output_file: JSONL output file path
        """
        # Clear the output file first
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                pass  # Just create/clear the file
        except Exception as e:
            print(f"Error creating output file: {str(e)}")
            return
        
        successful_count = 0
        
        for i, item in enumerate(tqdm(data_list, desc="Processing items")):
            print(f"Processing item {i+1}/{len(data_list)}...")
            
            # Extract instruction, input, and expected output
            instruction = item.get('instruction', '')
            input_smiles = item.get('input', '')
            expected_output = item.get('output', '')
            
            # Create prompt
            prompt = self.create_prompt(instruction, input_smiles)
            
            # Get model prediction
            prediction = self.get_model_response(instruction, input_smiles)
            
            # Create result entry
            result = {
                "prompt": prompt,
                "predict": prediction,
                "label": expected_output
            }
            
            # Save immediately to JSONL
            try:
                with open(output_file, 'a', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
                successful_count += 1
                print(f"✓ Saved item {i+1} to {output_file}")
            except Exception as e:
                print(f"Error saving item {i+1}: {str(e)}")
            
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
        
        print(f"\n🎉 Successfully processed and saved {successful_count}/{len(data_list)} items to {output_file}")
        return successful_count


def main():
    """
    Main function to process molecule inference
    """
    print("=== 分子推理处理器 / Molecule Inference Processor ===")
    print("配置选项 / Configuration options:")
    print("1. 设置并行工作线程数 / Set number of parallel workers")
    
    # Get max_workers from user
    while True:
        try:
            max_workers_input = input("输入并行工作线程数 (1-20, 推荐5-8) / Enter number of parallel workers (1-20, recommended 5-8) [default: 5]: ").strip()
            if not max_workers_input:
                max_workers = 5
                break
            max_workers = int(max_workers_input)
            if 1 <= max_workers <= 20:
                break
            else:
                print("请输入1-20之间的数字 / Please enter a number between 1-20")
        except ValueError:
            print("请输入有效数字 / Please enter a valid number")
    
    processor = MoleculeInferenceProcessor(max_workers=max_workers)
    
    # File paths
    input_file = "/mnt/ceph/users/zlu10/llm/MolReasoner/LLaMA-Factory/data/tox_test.json"
    output_json_file = "../output/4o_tox.json"
    output_jsonl_file = "../output/4o_tox.jsonl"
    
    print(f"\n使用 {max_workers} 个并行工作线程 / Using {max_workers} parallel workers")
    print("选择处理模式 (Choose processing mode):")
    print("1. 🚀 并行处理全部数据并保存为单个JSON文件 (Parallel process all data, save as single JSON)")
    print("2. 🚀 并行处理全部数据并逐个保存为JSONL (Parallel process all data, save streaming as JSONL)")
    print("3. 🐌 顺序处理全部数据并保存为单个JSON文件 (Sequential process all data, save as single JSON)")
    print("4. 🐌 顺序处理全部数据并逐个保存为JSONL (Sequential process all data, save streaming as JSONL)")
    print("5. 🧪 处理前5个样本测试 (Process first 5 samples for testing)")
    print("6. 取消 (Cancel)")
    
    choice = input("请输入选择 (Enter choice) (1/2/3/4/5/6): ")
    
    # Load test data
    test_data = processor.load_test_data(input_file)
    if not test_data:
        print("No data loaded. Exiting.")
        return
    
    if choice == '1':
        print(f"🚀 将并行处理全部 {len(test_data)} 个样本并保存到: {output_json_file}")
        confirm = input("确认要处理全部数据吗？(Confirm processing all data?) (y/n): ")
        if confirm.lower() in ['y', 'yes']:
            processor.process_inference_parallel(test_data, output_json_file)
        else:
            print("已取消处理 (Processing cancelled).")
    
    elif choice == '2':
        print(f"🚀 将并行处理全部 {len(test_data)} 个样本并保存到: {output_jsonl_file}")
        confirm = input("确认要处理全部数据吗？每个结果会立即保存。(Confirm processing all data? Each result will be saved immediately.) (y/n): ")
        if confirm.lower() in ['y', 'yes']:
            processor.process_inference_parallel_streaming(test_data, output_jsonl_file)
        else:
            print("已取消处理 (Processing cancelled).")
    
    elif choice == '3':
        print(f"🐌 将顺序处理全部 {len(test_data)} 个样本并保存到: {output_json_file}")
        confirm = input("确认要处理全部数据吗？(Confirm processing all data?) (y/n): ")
        if confirm.lower() in ['y', 'yes']:
            processor.process_inference_batch(test_data, output_json_file)
        else:
            print("已取消处理 (Processing cancelled).")
    
    elif choice == '4':
        print(f"🐌 将顺序处理全部 {len(test_data)} 个样本并保存到: {output_jsonl_file}")
        confirm = input("确认要处理全部数据吗？每个结果会立即保存。(Confirm processing all data? Each result will be saved immediately.) (y/n): ")
        if confirm.lower() in ['y', 'yes']:
            processor.process_inference_streaming(test_data, output_jsonl_file)
        else:
            print("已取消处理 (Processing cancelled).")
    
    elif choice == '5':
        sample_data = test_data[:5]
        sample_output_file = "molecule_inference_sample.json"
        print(f"🧪 将处理前5个样本并保存到: {sample_output_file}")
        processor.process_inference_batch(sample_data, sample_output_file)
    
    elif choice == '6':
        print("Processing cancelled.")
    
    else:
        print("无效选择，程序退出 (Invalid choice, exiting).")


if __name__ == "__main__":
    main()