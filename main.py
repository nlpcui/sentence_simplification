from models import *
from evaluator import *
from utils import read_data
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--paths', type=str, default='config/euler_config.json')  # local/config
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_rate', type=float, default='0.03')
    parser.add_argument('--model_save_path', type=str, default='./bart_base_simplifier.pt')
    parser.add_argument('--result_save_path', type=str, default='./result.csv')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--gpu_id', type=int, default=3)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='facebook/bart-base')
    parser.add_argument('--max_examples', type=int, default=1000)

    parser.add_argument('--job', type=str, default='inference')
    args = parser.parse_args()

    set_seed(args.random_seed)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    with open(args.paths, 'r') as fp:
        paths = json.load(fp)

    device = torch.device('cuda:{}'.format(args.gpu_id)) if args.gpu_id >= 0 else torch.device('cpu')

    if args.job == 'train':
        logging.info('train a bart model ...')
        original, simplified = read_data(paths['gpt3_simplified_manifesto_2000'], args.max_examples)
        assert len(original) == len(simplified)
        train_bart(args, original, simplified, paths['model_bart_manifesto_gpt3_2k'], device)

    elif args.job == 'inference':
        logging.info('simplify texts ...')
        original, simplified = read_data(paths['manifesto_complex'])
        assert len(original) == len(simplified)
        inference(args, original, simplified, paths['model_bart_manifesto_gpt3_2k'], paths['manifesto_simplified_by_bart_gpt3_2000'], device)

    elif args.job == 'evaluate':
        logging.info('evaluate simplification ...')
        original, simplified_reference = read_data(paths['minwiki_test'])
        original_, simplified_model = read_data(result_file=paths['minwiki_test_simplified_by_bart_minwiki_gpt3_1k'])
        assert original == original_

        ts_evaluator = TextSimplificationEvaluator(original=original, reference=simplified_reference,
                                                   simplified=simplified_model)
        model_result = ts_evaluator.compute_metrics()
        reference_result = ts_evaluator.compute_reference_metrics()

        print(model_result)
        print(reference_result)
