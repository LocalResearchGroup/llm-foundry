from math_verify import parse, verify
import torch
import logging
from llmfoundry.eval.metrics.nlp import InContextLearningMetric

log = logging.getLogger(__name__)

class MathVerifyMetric(InContextLearningMetric):
    """Metric that uses Math-Verify for robust mathematical expression evaluation."""
    
    # Make torchmetrics call update only once
    full_state_update = False
    
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('correct', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.metric_result_dict = {
            'cleaned_output': [],
            'original_label': [],
            'cleaned_label': [],
            'result': [],
        }
        
    def update(self, batch: dict, outputs: list[str], labels: list[list[str]]) -> dict:
        """Update the state with new predictions and targets."""
        metric_result_dict = self.metric_result_dict.copy()
        
        for pred, target_list in zip(outputs, labels):
            try:
                parsed_pred = parse(pred)
                for target in target_list:
                    parsed_target = parse(target)
                    if verify(parsed_target, parsed_pred):
                        self.correct += 1  # type: ignore
                        metric_result_dict['result'].append(1)
                        break
                else:
                    metric_result_dict['result'].append(0)
            except Exception as e:
                # Log error but count as incorrect
                log.warning(f"Failed to evaluate prediction '{pred}' against targets '{target_list}': {str(e)}")
                metric_result_dict['result'].append(0)
                
            metric_result_dict['cleaned_output'].append(pred)
            metric_result_dict['original_label'].append(target_list)
            metric_result_dict['cleaned_label'].append(target_list)
            self.total += 1  # type: ignore
            
        return metric_result_dict
            
    def compute(self) -> torch.Tensor:
        """Aggregate state over all processes and compute the metric."""
        return 100 * (self.correct / self.total)  # type: ignore 
