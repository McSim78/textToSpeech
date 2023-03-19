import argparse
import os
import torch

SPEAKER = ''
LANG = ''
INPUT_FILE = ''
INPUT_TEXT = ''
NUM_THREADS = 8  # number of threads for CPU inference


def get_speakers(lang: str = 'ru'):
    prefix = "en_"
    nums = range(0, 118)
    names_ru = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']
    names_en = [f"{prefix}{num}" for num in nums]
    names_en.append('random')

    if lang == 'ru':
        return names_ru
    elif lang == 'en':
        return names_en
    else:
        return names_ru + names_en


def text_to_speach(lang: str = 'ru', text: str = '', speaker: str = 'baya', file: str = 'input.txt'):
    device = torch.device('cpu')
    torch.set_num_threads(NUM_THREADS)

    if speaker not in get_speakers(lang):
        raise Exception(f'Для языка {lang} указан неверный голос "{speaker}".\n'
                        f'Доступны следующие голоса: {get_speakers(lang)}')

    if text:
        input_text = text
    elif file:
        os.path.isfile(file)
        with open(file, 'r') as f:
            input_text = f.read()
    else:
        raise Exception('Не указан текст для озвучки')

    local_file = ''
    sample_rate = 48000

    if lang == 'ru':
        local_file = 'v3_1_ru.pt'
    elif lang == 'en':
        local_file = 'v3_en.pt'

    if not os.path.isfile(local_file):
        print(f'Скачиваю файл модели {local_file}...')
        torch.hub.download_url_to_file(f'https://models.silero.ai/models/tts/{lang}/{local_file}',
                                       local_file)

    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)

    audio_paths = model.save_wav(text=input_text, speaker=speaker, sample_rate=sample_rate)
    if audio_paths:
        print(f'Озвучка сохранена в файле: {audio_paths}')
    else:
        print('Не удалось сделать озвучку. Возможно, указаны неверные параметры.')


def main():
    text_to_speach(lang=LANG, text=INPUT_TEXT, speaker=SPEAKER, file=INPUT_FILE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Озвучка текста')
    parser.add_argument('--lang', type=str, default='ru', help='Язык, в который нужно сделать озвучку (ru, en)',
                        choices=['ru', 'en'], required=True)
    parser.add_argument('--text', type=str, help='Текст для озвучки')
    parser.add_argument('--file', type=str, default='input.txt', help='Файл с текстом для озвучки')
    parser.add_argument('--speaker', type=str,
                        help='Голос, который нужно использовать. Для русского языка: aidar, baya, kseniya, xenia, '
                             'eugene, random. Для английского языка: en_0, en_1, ..., en_117, random')
    args = parser.parse_args()
    # Check arguments
    LANG = args.lang
    if LANG not in ['ru', 'en']:
        print('Неверный язык. Доступные языки: ru, en')
        exit(1)
    # Check speaker
    SPEAKER = args.speaker
    if SPEAKER not in get_speakers():
        print('Неверный голос для озвучки')
        exit(1)
    # Check text or file
    INPUT_FILE = args.file
    INPUT_TEXT = args.text
    if INPUT_TEXT is None and INPUT_FILE is None:
        print('Необходимо указать текст или файл с текстом для озвучки')
        exit(1)
    # Run
    try:
        main()
    except Exception as e:
        print(e)
        exit(1)

