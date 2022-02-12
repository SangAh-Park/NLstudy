from gensim.models import Word2Vec

#모델 로딩
model = Word2Vec.load('단어 임베딩/nvmc.model')
print("corpus_total_words:", model.corpus_total_words)

#'사랑'이란 단어로 생성한 단어 임베딩 벡터
print('사랑:', model.wv['사랑'])

#단어 유사도 계산하기
print("일요일 = 월요일\t", model.wv.similarity(w1='일요일', w2='월요일'))