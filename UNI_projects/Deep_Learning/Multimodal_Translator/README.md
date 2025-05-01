# Sentiment Analysis

## Introduction

This project focuses on building a multimodal translator using several deep learning models from HuggingFace and Gradio for the GUI.

## Features

This translator allows for translation from and to 20 different languages

It accepts both textual and audio inputs and the language detection is taken care of automatically by the application.

The application produces both textual and audio outputs in the desired language.

## Architecture

Gradio is used for the GUI and manages the gathering of the inputs and dispaying the outputs after the computations are done.

I used several AI models from HuggingFace to build the translator, these are:
- [OpenAI: Whisper Large V3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo): This model is used to transcribe audio to text if the input is audio.
- [Papluca: XLM-Roberta Base Language Detection](https://huggingface.co/papluca/xlm-roberta-base-language-detection): This model is used to detect the language of the input text.
- [Facebook: NLLB-200 Distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M): This is the model used to translate from one language to another.
- [Suno: Bark](https://huggingface.co/suno/bark): This is the model used to synthesize audio from text.

Basically the application works as follows:

- The user chooses whether to translate from text or audio with a radio button.
- The user provides the input text or audio.
- The user chooses the language to translate to.
- The application uses the Whisper model to transcribe the audio to text if the input is audio.
- The application uses the XLM-Roberta model to detect the language of the input text.
- The application uses the NLLB-200 Distilled 600M model to translate from the detected language to the chosen language.
- The application uses the Bark model to synthesize audio from the translated text.
- The application displays the translated text and audio.

These are some of the results the model obtains:

![Results with textual input](./text_input.mp4)

![Results with audio input](./audio_input.mp4)