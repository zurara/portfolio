# folk_winter

用 Python（Pillow + NumPy）程序化生成的冬日村庄，素人画 / 民间绘画风格：灰绿色天空、锯齿雪山、谷底雾气、散落的雪顶小屋、湖面滑冰人群、篝火、斜坡上的栅栏和行人、前景高大的秃树。

```bash
pip install pillow numpy
python folk_winter.py                 # 输出 folk_winter.png
python folk_winter.py --seed 42 --out v2.png   # 换一个随机种子
```

- 以 2 倍分辨率绘制后缩小，线条抗锯齿。
- 近大远小：`perspective(y)` 按离地平线的距离缩放房屋和人物。
- 最后叠一层噪点和横向纹理，模拟画布质感。
- 调颜色改文件顶部的常量；调构图改 `compose()` 里的湖泊、人群、树木坐标。

## 印象派版本：`impressionist.py`

把 `folk_winter.png` 當底圖，用約 10 萬筆短筆觸重畫一遍。

```bash
pip install pillow numpy scipy
python impressionist.py --src folk_winter.png --out impressionist.png   # 約 15 秒
```

1. **調色**（`monet_grade`）：暗部往藍紫偏，亮部往米黃偏，去掉純黑；雪地加大片的淡紫、淡橘色塊；天空調亮，混入玫瑰色和赭色；冰面混入藍色。
2. **分層筆觸**（`paint`，Hertzmann 1998 的做法）：筆刷半徑 22 → 12 → 7 → 4 px。第一層鋪滿整張畫，之後每層只畫在「畫布和參考圖差異大」的地方。
3. **筆觸方向**：用結構張量算出邊緣方向，筆觸沿邊緣走；沒有明顯邊緣的地方（雪地、天空）大致橫向，帶隨機偏轉。
4. **筆觸本身**（`paint_stroke`）：每一筆是一束平行的刷毛線，兩側的刷毛較短；顏色做色相、飽和度、明度的隨機偏移（破色）。
5. **收尾**（`impasto`）：用亮度細節模擬顏料凸起的光影，再加一點布紋噪點。

調參：`BRUSHES` 改筆刷大小，`THRESHOLD` 改細筆觸補多少，`paint_stroke` 裡的 `length` 改筆觸長短。
