from dataset_word import WordDataset
from dataset_label import LabelDataset
from dataset_img import ImageDataset
from dataset_img_region import ImageRegionDataset
from dataset_visual_question import VisualQuestionDataset


def build_datasets(**params):
    encoder_input_filename = params['encoder_input_filename']
    decoder_input_filename = params['decoder_input_filename']
    classifier_input_filename = params['classifier_input_filename']
    encoder_datatype = params['encoder_datatype']
    decoder_datatype = params['decoder_datatype']
    classifier_datatype = params['classifier_datatype']
    enable_classifier = params['enable_classifier']

    print("Loading datasets...")
    datasets = {}
    datasets['encoder'] = find_dataset(encoder_input_filename, encoder_datatype)(encoder_input_filename, is_encoder=True, **params)
    datasets['decoder'] = find_dataset(decoder_input_filename, decoder_datatype)(decoder_input_filename, is_encoder=False, **params)
    if enable_classifier:
        datasets['classifier'] = find_dataset(classifier_input_filename, classifier_datatype)(decoder_input_filename, is_encoder=False, **params)
    return datasets


def find_dataset(input_filename, dataset_type=None):
    types = {
        'img': ImageDataset,
        'bbox': ImageRegionDataset,
        'vq': VisualQuestionDataset,
        'txt': WordDataset,
        'lab': LabelDataset,
    }
    if dataset_type:
        return types[dataset_type]
    # If no dataset type is specified, infer based on file extension
    ext = input_filename.split('.')[-1]
    return types.get(ext, WordDataset)
