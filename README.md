## [모델링]

### Semantic Seimilarity(의미 유사성)
   - 의미 유사성은 텍스트 간 유사도를 판별하는 task를 의미한다.
   <br>
  
### Contextual Word Embedding(BERT) 
  
- 의미 유사성은 언제 or 왜 필요할까??
   - 스택오버플로우, Quora, Yahoo Answer 등의 QA 플랫폼은 수많은 사용자가 정보를 얻고자 질문을 게시하고, 또 다른 사용자가 올린 질문에 대해 답을 남기는 구조를 가진다. 
   <br>
   - 특정 정보가 필요한 사용자는 이미 관련된 질문과 답 pair가 있다면 새로운 질문을 굳이 게시할 필요가 없어 굳이 기다릴 필요 없다.
   <br>
   - 이러한 텍스트를 매칭은, `텍스트 임베딩`. 즉, 텍스트를 그대로 사용하지 않고 벡터로 변환하여 벡터간 연산을 통해 유사한 텍스트를 매칭한다.
      - 대표적으로 아래 세 가지 방법을 사용
   
    |method|특징|example|
    |:---:|:---:|:---:|
    |Traditional Word Embedding|단어 빈도수 기반 표현|BoW, tf-idf...|
    |Static Word Embedding|학습을 통해 단어를 고정된 임베딩으로 변환|Word2Vec, GloVe, FastText...|
    |`Contextual Word Embedding`|문맥을 반영한 임베딩 생성|BERT, RoBERTa...|
   
   - Contextual 방식은 `문맥`을 반영할 수 있다는 점에서 나머지 두 방법보다 의미 유사성을 탐지하는데 훨씬 유리함
      - `다의어` 혹은 `동음이의어`가 등장하는 경우, 빈도 기반(traditional) or 고정된 벡터로 변환(static)한다면 의미 유사성 측정에 제약이 발생
   

- 따라서 제안 모델에서는 self-attention을 사용하는 BERT를 선택하였다.


<br>

### Topic Model(VAE-NTM)


- 의미 유사성을 측정하기 위해서는 최대한 많은 정보를 사용하는 것이 중요하다.

- BERT의 경우 문맥 수준의 임베딩만 도출할 뿐 Topic-level 정보를 추출할 수 없다.

- 따라서, 텍스트를 임베딩으로 변환할 때 최대한 많은 정보를 활용하기 위해 BERT의 임베딩과 토픽 모델의 문서-토픽 분포(Document-Topic Distribution)을 사용하여 결합(Concatenate)한다면 보다 풍부한 정보를 갖는 임베딩을 생성할 수 있다. 

<br>

---

## Project Purpose 
- 본 프로젝트의 궁극적인 목표는 의미 유사도를 측정하는 데 문맥 정보 + 토픽 정보를 모두 이용한 Topic Enhanced BERT(TE-BERT)를 제안하여 최종 임베딩 Representation 성능을 높여 유사도 측정 정확도를 높이고자 함

<br>

## TE-BERT Structure
   

![](https://velog.velcdn.com/images/yjhut/post/7e5b943b-eec5-4c79-b5ea-1e314c8e9616/image.png)


- `특징 01. 인코더 구조`

   - Siamese Network(샴 네트워크)를 사용하여 각 텍스트의 문맥 임베딩과 토픽 벡터를 생성
      - 샴 네트워크란, 두 텍스트를 임베딩으로 생성하는 모델(여기서는 BERT와 VAE-NTM)이 모두 동일한 가중치를 갖는 구조를 의미 
      
      <br>
      
 
      
      
    >✋ 샴 네트워크를 사용하는 이유 ✋
    > - BERT의 경우, 모델 학습 시 `NSP(Next Sentence Prediction)`를 수행하여 두 문장 간 관계를 이해할 수 있도록 학습되어,
    > - BERT는 두 텍스트를 매칭할 때 `Cross Encoder` 혹은 `Bi-encoder`를 사용할 수 있다.
    > - 결과부터 말하면 샴 네트워크는 Bi encoder 구조를 사용하며, QA 텍스트 매칭에 적합하다.
    > <br>
    >
    >> 💡 Cross Encoder
    >> - 크로스 인코더란, 두 문장을 BERT에 한 번에 입력하여 의미 관계를 추론하는 구조를 사용
    >> - Cross Encoder 구조는 아래 그림과 같다.
    >>   
    >> ![](https://velog.velcdn.com/images/yjhut/post/87af1f9d-8e2b-416b-b6a2-174d879d05e3/image.png) | Devlin et al., (2019)
    >
    >> 💡 Bi Encoder
    >> - 바이 인코더란, 두 문장을 각 따로 입력하는 방법으로, 생성된 두 임베딩을 concat하여 최종 예측에 사용한다.
    >> - Bi Encoder 구조는 아래 그림과 같다.
    >>   
    >>  ![](https://velog.velcdn.com/images/yjhut/post/e8775ac7-351c-4eed-9ff0-7a139c52286b/image.png) | Reimers, N., & Gurevych, I. (2019)
    >
    >>  🚀 두 구조의 차이점 🚀
    >> - 바이 인코더와 크로스 인코더는 `속도`와 `정확도`에서 명확한 차이를 보인다.
    >> - 크로스 인코더의 경우, 두 문장 간 어텐션을 수행하여 바이 인코더에 비해 유사도 측정의 정확도가 높은 장점이 있다.
    >> - 하지만 본 프로젝트 처럼 QA에서 실시간 유사 질문을 찾기 위한 작업에서 `크로스 인코더는 연산량이 매우 큰 치명적인 단점`이 존재한다.
    >>     - ex) 10,000개 문장에서 유사한 텍스트를 찾을 때, 
    >>          - `크로스 인코더는 49,995,000번 계산 필요하고 GPU V100 기준 약 65시간 소요`
    >>          - 반면에 `바이 인코더는 임베딩 간 연산만 수행, 수 초내 해결`
  
 
  
- 따라서, `QA에서 유사한 텍스트 매칭 위한 목적의 모델이라면 크로스 인코더보다 바이 인코더 구조를 사용하는 게 훨씬 유용`하다고 할 수 있다.

---

- `특징 02. 풀링`

   - 여러 개 단어 임베딩을 문장 임베딩으로 표현하기 위해 `CLS Token`을 사용하거나 모든 단어 임베딩 피처별 평균을 사용하는 `평균 풀링(Mean Pooling)` 사용
   <br>
      
- `특징 03. VAE-based Neural Topic Model(VAE-NTM) 사용`

   - 여러 토픽 모델 중 VAE 기반의 토픽 모델의 장점은
       - 신경망 구조를 사용한 학습 가능
       - 병렬 처리 가능(기존 LDA와 같은 확률 기반의 토픽 모델보다 속도 빠름)
       - 텍스트 길이가 짧은 경우에도 LDA, NMF와 같은 전통적인 토픽 모델보다 Sparsity에 강건
  
  <br> 
  
---  
  
- Datasets

   - `Classification`
      - `PAWS(Paraphrase Adversaries With Sentence Shuffling)`
        - Train :49,4K
        - Validation : 2K
        - Test : 2K

        
 <br>      
 
 
- Metrics
  - `Accuracy`
  - `F1-score`
  - `Recall`
  - `Precision`

   
 <br> 

- 비교 모델 
   1. Vanila BERT(CLS Token)
   2. Vanila BERT(Mean Pooling)
   3. TE-BERT(CLS Token) -> `proposed`
   4. TE-BERT(Mean Pooling) -> `proposed`

---
   
## Expeiments & Results

### 텍스트 전처리(preprocessing)

- Text의 경우, 사용하는 모델에 따라 다른 전처리가 필요
   - BERT와 같은 어텐션 기반의 모델은 Stopwords, Lemmatization, Pos_tagging을 적용하면 오히려 단어간 문맥 파악에 부정적인 영향을 끼쳐, `BERT에 사용할 텍스트는 tag 제거, 소문자 변환(lower) 만 수행`
    <br> 
   - VAE-NTM은 별도 전처리를 추가로 수행하여 BoW를 구성하였다. 
      - `불용어 제거` 
      - `품사 태깅(Pos_tagging)`
      - `표제어 추출(Lemmatization)`

<br>

- 또한 BoW를 생성할 vectorizer 학습 시 너무 적은 빈도를 갖는 단어는 제거하기 위해 최소 N개 문서에 등장해야하는 `min_df`를 사용하였다.
   - 0~1로 지정하면 (전체 문서 개수 * 비율)번 이상 등장한 단어만 사용하고,
   - 자연수(N)로 지정한 경우 N번 이상 등장한 단어만 사용한다.

<br>

- 실험에 사용한 가중치 초기화 시드는 1~1000에서 랜덤 추출한 `968`을 사용한다.


<br>  

***

### 01. Vanila BERT(CLS Token, Mean pooling)

- 비교 모델 vanila BERT의 모델/분류기(단일 레이어) 학습률, 에포크, patience, 배치 크기는 아래와 같다.

  |Params|value|
  |:---:|:---:|
  |BERT_LR|1e-5|
  |Classifier_LR|1e-4|
  |Epochs|30|
  |Patience (early_stop)|5|
  |delta (early_stop)|0.005|
  |Batch_size|16|
  

<br> 
<br> 


### 02. TE-BERT(CLS Token, Mean pooling)

- BERT, 분류기 구조 학습률은 vanila 모델과 동일하게 사용하였고, VAE-NTM의 경우, 세 가지 특징을 선정하였다.

   `1. 토픽 개수`
   
   `2. 사전, 사후 분포로 가정할 분포`
   
   `3. 사전 분포의 alpha`

<br>

> 각 특징은 본 모델에서 사용한 VAE-NTM은 사전/사후 분포를 `디리클레분포(Dirichlet Distribution)`로 가정하였음을 전제로 한다.



---


- `토픽 개수`
   - 토픽 모델에서 토픽 개수는 매우 중요한 요소
   - VAE는 latent variable(잠재 변수)을 토픽 정보로 사용할 수 있다. 
   - 따라서 본 모델은 잠재 변수를 `문서-토픽 분포` 사용하였다.
 
<br> 

- `사전, 사후 분포로 가정할 분포`
   - 앞서 언급했듯, 디리클레 분포 사용
   - `잠재 변수를 문서 토픽 분포로 사용하기 위해 전체 합이 1이 되도록 만들어야 하며, 이러한 특징을 보존하기 위해 디리클레 분포를 선택`하였다.

<br> 

- `사전 분포 alpha`
   - 디리클레 분포의 경우 인코더를 사용하여 `alpha`를 추출하고, 문서-토픽 분포 샘플링
 
 
> - `토픽 개수`와 `사전분포 alpha`는 LDA를 이용하여 선정
> - alpha 별 coherence가 가장 높은 토픽 개수 조합 생성

> - 토픽 개수, alpha 후보
>    - `alpha = [0.1, 0.5, 1, 5, 10]`
>    - `topic_num = range(50,500,50) `

---


- LDA 실험 결과
   -  50~500개(step:50)에서 탐색하였다. 
![](https://velog.velcdn.com/images/yjhut/post/81ce6121-b1e8-4d27-b81b-ee9a6530da32/image.png)
   

- 그 결과, 아래와 같이 5개의 후보군이 정해졌다.

  |alpha|num_topic|
  |:---:|:---:|
  |0.1|100|
  |0.5|300|
  |`1`|`200`|
  |5|150|
  |10|100|

- 5개 후보 중 `alpha : 1, nnum_topic : 200`  조합이 TE-BERT에서 가장 좋은 성능을 보여 최종 파라미터로 선정하였다.


<br>


---

### Results

- Vanila BERT, TE-BERT 각 CLS token, Mean pooling을 사용한 결과는 아래와 같다.

<br>

  <table>
    <tr>
      <td align="center">Model</td>
      <td align="center">Accuracy</td>
      <td align="center">f1 score</td>
    </tr>
    <tr>
      <td colspan="3">CLS token</td>
    </tr>
    <tr>
      <td>Vanila BERT</td>
      <td align="center">83</td>
      <td align="center">83.7</td>
    </tr>
    <tr>
      <td>TE-BERT</td>
      <td align="center">84.5<br><small>(+ 1.8%)</small></td>
      <td align="center">85<br><small>(+ 1.55%)</small></td>
    </tr>
    <tr>
      <td colspan="3">Mean pooling</td>
    </tr>
    <tr>
      <td>Vanila BERT</td>
      <td align="center">86.2</td>
      <td align="center">86.1</td>
    </tr>
    <tr style="background-color: #696969;">
      <td>TE-BERT</td>
      <td align="center"><b>87.4</b><br><small><b>(+ 1.39%)</b></small></td>
      <td align="center"><b>87.1</b><br><small><b>(+ 1.16%)</b></small></td>
    </tr>
  </table>
| () -> Vanila 모델 대비 상대 향상률


<br>
<br>

- 실험한 네 모델 중 `VAE-NTM 결합, 평균 풀링을 사용한 TE-BERT가 성능`이 가장 좋았으며

- 같은 풀링 방법 내에서 비교했을 때도 모두 개선된 결과를 보였다.
   - `CLS token`의 경우 `accuracy: 1.8%`, `f1 score: 1.55%` 향상
   - `Mean pooling`의 경우 `accuracy: 1.39%`, `f1 score: 1.16%` 향상 


---


## 정리

- 유사도 측정을 위한 텍스트 임베딩 생성 시, 문맥적 정보와 토픽 정보를 모두 포함한 TE-BERT 모델 성능이 가장 좋았으며

- 현재 VAE의 인코더 디코더는 뉴럴 네트워크로 구성하였으나, 특징 추출을 위해 다른 모델(ex- CNN, RNN) 등 추가 모델을 결합하여 새로운 구조의 VAE를 고려할 수도 있을 것 같다.
