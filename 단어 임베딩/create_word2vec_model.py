from gensim.models import Word2Vec
from konlpy.tag import Komoran 
import time

# 네이버 영화 데이터 읽어오기 (전처리)
def read_review_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data

#학습 시간 측정 시작
start = time.time()

#리뷰 파일 읽어오기
print('1) 말뭉치 데이터 읽기 시작')
review_data = read_review_data('단어 임베딩/ratings.txt')
print(len(review_data))
print('1) 말뭉치 데이터 읽기 완료:', time.time()-start) #읽는 데 걸린 시간

#문장에서 명사만 추출해 학습 데이터로 만들기
print('2) 형태소에서 명사만 추출 시작')
komoran = Komoran()
docs = [komoran.nouns(sentence[1]) for sentence in review_data]
print('2) 형태소에서 명사만 추출 완료:', time.time()-start)

#word2vec 모델 학습
print('3) word2vec 모델 학습 시작')
model = Word2Vec(sentences=docs, window=4, min_count=2, sg=1) 
#size: 벡터 차원의 크기. hs: softmax 사용 시 1. min_count: 단어 최소 빈도 수 제한. sg: 0은 CBOW, 1은 skip-gram.
print('3) word2vec 모델 학습 완료:', time.time()-start)

#모델 저장
print('4) 학습된 모델 저장 시작')
model.save('nvmc.model')
print('4) 학습된 모델 저장 완료:', time.time()-start)

#학습된 말뭉치 수, 코퍼스 내 전체 단어 수
print("corpus_count:", model.corpus_count)
print("corpus_total_words", model.corpus_total_words)