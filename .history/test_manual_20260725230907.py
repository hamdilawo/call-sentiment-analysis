from app.asr.transcriber import get_transcriber
from app.sentiment.analyzer import get_sentiment_analyzer
from app.preprocessing.audio_preprocessing import preprocess_audio

transcriber = get_transcriber()
analyzer = get_sentiment_analyzer()

for f in ["audio_samples/positif.wav", "audio_samples/negatif.wav", "audio_samples/neutre.wav"]:
    signal = preprocess_audio(f)
    texte = transcriber.transcribe(signal)
    result = analyzer.analyze(texte)
    print(f"{f} -> \"{texte}\" -> {result['sentiment']} ({result['confidence']})")