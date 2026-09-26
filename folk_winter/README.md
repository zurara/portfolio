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
