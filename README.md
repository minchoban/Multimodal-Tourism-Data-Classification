# Multimodal-Tourism-Data-Classification
2022 관광데이터 AI 경진대회에서 사용한 코드를 정리한 Repository입니다.

POI 데이터를 이용하여 카테고리를 분류하는 모델을 구현하였습니다. 관광데이터의 이미지와 이에 따른 텍스트 설명을 VIT/RoBerTa를 이용한 멀티모달로 분류를 진행하였습니다.

데이터는 아래 링크에서 확인하실 수 있습니다.
https://dacon.io/competitions/official/235978/data

핵심적인 방법은 확률적으로 텍스트 데이터를 증강한 것입니다.
텍스트 데이터를 증강하는 방법은 Easy Data Augmentation을 사용하였습니다.

데이터의 특성상 128개의 많은 카테고리를 분류해야하며, 각 카테고리의 개수는 매우 불균형한 것을 확인할 수 있습니다.

![image](https://github.com/minchoban/Multimodal-Tourism-Data-Classification/assets/99804394/b611244c-4d2c-4092-b9b8-a06c12a07c42)
제일 데이터가 많은 카테고리는 3000개가 넘어가는 반면, 가장 적은 카테고리는 2개밖에 없습니다.

데이터를 일괄적으로 증강을 하는 것이 아니라, 데이터의 개수에 따라서 데이터 증강을 다르게 하는 방법을 선택하였습니다.
데이터 개수를 기준으로 threshold를 잡아 데이터 증강을 진행할 경우, 역전 현상이 발생할 수 있습니다.

예를 들어 임계점을 200개로 잡고 200개 이상인 경우 데이터 증강을 2배, 200개 미만인 경우 데이터 증강을 3배 진행하는 경우
201개인 데이터는 402개가 되는 반면, 199개인 데이터는 597개가 되어 데이터의 분포가 바뀌게 됩니다.

이를 해결하기 위해 확률적인 증강 방법을 고안하였습니다.

데이터의 개수가 많을수록 적게, 적을수록 많게 데이터를 증강하는 방법입니다.

* 카테고리의 데이터 개수를 전체 데이터 개수로 나눈 후 역수를 취합니다. 
* 이후 루트를 씌워 분산을 줄이고 이를 min_max scaling을 진행하여 확률값으로 사용합니다.
* 각 관찰 값에 대해 binomial distribution을 통해 0, 1 값을 추출하고
* 0이면 데이터 증강을 하지 않고 1이면 데이터를 증강하는 방법을 데이터를 증강하였습니다.

데이터 수가 적을수록 증강될 확률이 높아지기 때문에 전체 데이터의 분포를 해치지 않고 증강할 수 있습니다.
