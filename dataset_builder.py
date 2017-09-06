from dataset_word import WordDataset
from dataset_label import LabelDataset
from dataset_img import ImageDataset
from dataset_img_region import ImageRegionDataset
from dataset_visual_question import VisualQuestionDataset
from dataset_attribute import AttributeDataset


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
    # TODO: Replace hard coded train/test filenames with one-file dataset format
    datasets['encoder'] = dataset_by_type(encoder_datatype)(encoder_input_filename, is_encoder=True, **params)
    datasets['encoder_evaluate'] = dataset_by_type(encoder_datatype)(encoder_input_filename.replace('train', 'test'), is_encoder=True, **params)
    datasets['decoder'] = dataset_by_type(decoder_datatype)(decoder_input_filename, is_encoder=False, **params)
    datasets['decoder_evaluate'] = dataset_by_type(decoder_datatype)(decoder_input_filename.replace('train', 'test'), is_encoder=False, **params)
    if enable_classifier:
        datasets['classifier'] = dataset_by_type(classifier_datatype)(decoder_input_filename, is_encoder=False, **params)
        datasets['classifier_evaluate'] = dataset_by_type(classifier_datatype)(decoder_input_filename.replace('train', 'test'), is_encoder=False, **params)
    return datasets


def dataset_by_type(dataset_type):
    types = {
        'img': ImageDataset,
        'bbox': ImageRegionDataset,
        'vq': VisualQuestionDataset,
        'txt': WordDataset,
        'lab': LabelDataset,
        'att': AttributeDataset,
    }
    return types[dataset_type]
