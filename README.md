# Match-Analytics

Analytics de partida de futebol com pipeline **100% neuromórfica e feita do zero** — nenhum YOLO, nenhum modelo pré-treinado:

```
video -> DVS events -> detector SNN (conv-LIF, auto-labels)
      -> Kalman tracking -> kmeans = times -> homografia -> radar 2D
```

![Match analytics](demo/match.gif)

## Pipeline

| Etapa | Implementação |
|---|---|
| Eventos | DVS simulado (contraste log-temporal, ref adaptativa) |
| Detecção | **SNN detector** estilo CenterNet: conv-LIF → heatmap de centros + regressão de tamanho. Treinado com **auto-labels**: blobs de eventos viram ground truth, zero anotação manual |
| Tracking | **Filtro de Kalman do zero** (estado [x,y,vx,vy]) por jogador |
| Times | **kmeans do zero** na cor do uniforme (torso, sem verde do campo) |
| Radar | **Homografia calibrada** → posição real em metros no mini-mapa |
| Stats | distância percorrida por time, jogadores em campo, trilhas |

## Resultados

- 15–17 jogadores simultâneos detectados direto do stream de eventos
- Radar 2D com cores por time em tempo real
- Distância acumulada por time ao longo do clipe (~1–2.5 km por time em 60s de tracking)

## Uso

```bash
pip install -r requirements.txt

# 1) auto-labels de eventos
python -m src.auto_label raw/clip.mp4

# 2) treina o detector SNN nos pseudo-labels
python -m src.train_detector --epochs 40

# 3) calibra homografia: edite src->dst (metros) em calib_*.json

# 4) demo: partida + radar + stats
python -m src.demo_match raw/clip.mp4 --calib calib_m02.json --use-events
```

## Por que "do zero" importa aqui

O pipeline inteiro — neurônios LIF, surrogate gradient, heatmap decoder, Kalman, kmeans, homografia — é implementado sem frameworks de detecção. É a arquitetura que sistemas neuromórficos de esporte (Prophesee etc.) usam: eventos baratos, latência mínima, computação só onde há movimento.

Série neuromórfica:
[snn-ocr](https://github.com/vsousadelvek/snn-ocr) ·
[snn-gesture](https://github.com/vsousadelvek/snn-gesture) ·
[snn-tracker](https://github.com/vsousadelvek/snn-tracker) ·
[snn-anomaly](https://github.com/vsousadelvek/snn-anomaly) ·
[snn-forecast](https://github.com/vsousadelvek/snn-forecast)
