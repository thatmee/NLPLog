import time
import torch
import datasets
import torch.nn as nn
from nlplog import Config
from .scorer import calculate_score_by_log
from .train import valid_fn
from .utils import get_collate_and_bsz

def test(
        model,
        dataset:torch.utils.data.Dataset,
        test_data:datasets.Dataset,
        config:Config,
        load_best_model=True
    ):
    config.logger.info(f'================Test================')

    # dataset
    test_set = dataset(test_data, config)

    # dataloader
    collate_fn, batch_size = get_collate_and_bsz(dataset, config)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn)

    # criterion & device
    model.to(config.device)

    # load best model
    model_type = model.__class__.__name__
    if load_best_model:
        config.logger.info(f"load best model: {config.model_save_dir}{model_type}.pth")
        model.load_state_dict(torch.load(f"{config.model_save_dir}{model_type}.pth")['model'])

    # test and metrics
    _, predictions, logits = valid_fn(test_dataloader, model, config.device, len(test_data), config)
    report = calculate_score_by_log(predictions, test_data, config=config)


    if config.confusion_matrix_enable:
        config.logger.info(f"confusion_matrix:\n{report['confusion_matrix']}")
    for report_type in config.f1_report:
        precision, recall, f1 = report[report_type]['precision'], report[report_type]['recall'], report[report_type]['f1']
        config.logger.info(f"[{report_type}] precision: {precision:.4f} recall: {recall:.4f} f1: {f1:.4f}")

    # save predictions for further analysis
    time_str = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
    torch.save(
            obj={'predictions': predictions},
            f=f"{config.predictions_save_dir}{model_type}-{time_str}.pth"
    )

    return report