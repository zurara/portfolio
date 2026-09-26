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

## 照片轉油畫速寫：`paint_photo.py`

把一張照片（`field.jpg`，田野、晚霞、穀倉）畫成印象派的直接畫法（alla prima）油畫速寫。

```bash
python paint_photo.py --src field.jpg --out field_painted.png   # 約 15 秒
```

和 `impressionist.py` 的差別：

- **筆觸會彎、會停**：筆觸沿著形體方向生長，底下的顏色一變就停，所以一筆剛好蓋住一塊顏色。
- **沾了顏料的平頭筆**：每一筆由一束刷毛組成，部分刷毛沾到鄰近的顏色；筆頭是圓的，筆尾會乾掉、散開，露出暖色底（`GROUND`）。
- **分區畫法**（`region`）：天空和田地用長而平的筆觸，雲沿著雲形捲，樹林用短的豎向點筆，玉米田最後加細挑筆。
- **穀倉是手寫的筆觸**（`barn_strokes`）：屋頂三筆、山牆四五筆、側牆兩筆、屋簷和山牆邊的深色線各一筆、右邊紅屋頂三筆。自動筆觸層在穀倉範圍內不畫細筆，保留「一筆畫成」的樣子。
- **顏料厚度**：每一筆同時寫進一張高度圖，筆觸邊緣和起筆處比較厚；最後用這張高度圖打側光，薄的地方露出布紋。

穀倉和各區域的座標是按這張照片量的（照片座標，1932×2576）。換照片的話要重新量 `barn_strokes`、`barn_mask` 和檔案頂部的區域分界。
