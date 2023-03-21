# pipeline
related url: https://huggingface.co/tasks
## Audio (预装ffmpeg)
1. AudioClassificationPipeline
2. AutomaticSpeechRecognitionPipeline
## Computer vision
1. DepthEstimationPipeline
2. ImageClassificationPipeline
3. ImageSegmentationPipeline
4. ObjectDetectionPipeline
5. VideoClassificationPipeline
6. ZeroShotImageClassificationPipeline
7. ZeroShotObjectDetectionPipeline
## Natural Language Processing
1. ConversationalPipeline
2. FillMaskPipeline
3. NerPipeline
4. QuestionAnsweringPipeline
5. SummarizationPipeline
6. TableQuestionAnsweringPipeline
7. TextClassificationPipeline
8. TextGenerationPipeline
9. Text2TextGenerationPipeline
10. TokenClassificationPipeline
11. TranslationPipeline
12. ZeroShotClassificationPipeline
## Multimodal
1. DocumentQuestionAnsweringPipeline(预装tesseract)
2. FeatureExtractionPipeline
3. ImageToTextPipeline
4. VisualQuestionAnsweringPipeline

### notice：
You may need to modify Dockerfile content as needed

ps: tasks and their default models(pytorch):
``` 
{'audio-classification': ('superb/wav2vec2-base-superb-ks', '372e048'),
 'automatic-speech-recognition': ('facebook/wav2vec2-base-960h', '55bb623'),
 'conversational': ('microsoft/DialoGPT-medium', '8bada3b'),
 'depth-estimation': ('Intel/dpt-large', 'e93beec'),
 'document-question-answering': ('impira/layoutlm-document-qa', '52e01b3'),
 'feature-extraction': ('distilbert-base-cased', '935ac13'),
 'fill-mask': ('distilroberta-base', 'ec58a5b'),
 'image-classification': ('google/vit-base-patch16-224', '5dca96d'),
 'image-segmentation': ('facebook/detr-resnet-50-panoptic', 'fc15262'),
 'image-to-text': ('ydshieh/vit-gpt2-coco-en', '65636df'),
 'object-detection': ('facebook/detr-resnet-50', '2729413'),
 'question-answering': ('distilbert-base-cased-distilled-squad', '626af31'),
 'summarization': ('sshleifer/distilbart-cnn-12-6', 'a4f8f3e'),
 'table-question-answering': ('google/tapas-base-finetuned-wtq', '69ceee2'),
 'text-classification': ('distilbert-base-uncased-finetuned-sst-2-english',
                         'af0f99b'),
 'text-generation': ('gpt2', '6c0e608'),
 'text2text-generation': ('t5-base', '686f1db'),
 'token-classification': ('dbmdz/bert-large-cased-finetuned-conll03-english',
                          'f2482bf'),
 'translation_en_to_de': ('t5-base', '686f1db'),
 'translation_en_to_fr': ('t5-base', '686f1db'),
 'translation_en_to_ro': ('t5-base', '686f1db'),
 'video-classification': ('MCG-NJU/videomae-base-finetuned-kinetics',
                          '4800870'),
 'visual-question-answering': ('dandelin/vilt-b32-finetuned-vqa', '4355f59'),
 'zero-shot-classification': ('facebook/bart-large-mnli', 'c626438'),
 'zero-shot-image-classification': ('openai/clip-vit-base-patch32', 'f4881ba'),
 'zero-shot-object-detection': ('google/owlvit-base-patch32', '17740e1')}
```
