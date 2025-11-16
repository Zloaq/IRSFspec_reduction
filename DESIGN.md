## 1. 目的

## 2. 環境・依存


- 依存  
astropy.io.fits, numpy, sqlite3, subprocess, logging
- パス  
DB_PATH, RAW_DARK_PATH, DARK4LOCATE_DIR, RAID_PC/RAID_DIR, DST_DIR
- パラメータ:  
QUALITY_HIST_RANGE=(55000, 65536), BINS=1000


## 3. 関数定義



### 3.1 irsf_reduction.py

>**_load_fits**
>- FITSファイルを開いてヘッダとデータを返す。存在確認と例外処理あり。

>**db_search**
>- 引数 (conn, object_name, date_label)
>- object_name で frames テーブルを検索し、date_label 毎のベース名リスト辞書を返す。

>**do_scp**
>- 引数 (date_label, Number)
>- RAIDから指定ファイルをDST_DIRへscpコピー。失敗時は例外発生。

>**compute_hist**
>- 引数 (data, bins=BINS, rng=QUALITY_HIST_RANGE)
>- NaN除去後にヒストグラム計算。範囲未指定なら最大最小を使用。

>**hist_area_from_counts**
>- 引数 (hist, bin_edges, log_hist=False)
>- ヒストグラムの面積を計算。log_hist時は対数変換を適用。


>**compute_area**
>- 引数 (data, bins=BINS, rng=QUALITY_HIST_RANGE, log_hist=False)
>- ヒストグラム計算と面積算出をまとめて実行しスコアを返す。

>**quality_check**
>- 引数 (fitslist)
>- 各FITSの面積を計算し、エラー判定したファイルを除外して合格リストを返す。

>**classify_spec_location**
>- 引数 (fitsdict{No:fitslist}) って感じかな。
>- 中心 pix 同士の幅 >  典型的な像の広がり * 2 + buff  
最も明るいやつだけでいいか？だって quality check 通ってるんだもんね？  
でも誤検出で出ない可能性もあるかなあ。list にしとくかあ。  
>- spec_locator 通したら mask が返ってきます。

>**reject_saturation**
>- 引数 (fitslist)
>- リストの中を CDS30 の側から調べてサチってたら消したいけど、どこにスペクトルがあるかわかんねえな。　　
もう一回検出するのは馬鹿らしいけど、堅牢で分かりやすい処理って考えたら別にいいか。
spec 検出、mask して  < saturation 

>**search_combination_CDSnum**
>- 引数 ()
>- 全部入れてから組み合わせ見なきゃいけないよねえ。
とりあえず二つのリストだけにした。

>**reduction_main**
>- 引数 ()
>- まずは１天体,全日付の reductionを前提
db_search の出力を do_scp 用に加工する。



### 3.2 spec_locator.py




### 3.2 全体フロー（簡易）

```mermaid
flowchart TD

    DB[db_search] --> SCP[do_scp]
    SCP --> LD[load_fits]
    LD --> H1[compute_hist]
    H1 --> A1[hist_area]
    A1 --> CA[compute_area]
    CA --> QC[quality_check]
    QC --> NEXT[post steps]
```

### 3.3 注意・TODO
- `db_search` の `object_name` は現状 f-string で埋め込み。**SQLパラメータ化**を推奨（例: `WHERE object LIKE ?`）。
- `date_label` 条件の追加（期間/完全一致/前方一致など）。
- パス定数の**環境変数/引数化**（ハードコード脱却）。
- `quality_check` の閾値ロジック（`area > 0.0` の意味付け）を再検討。
- 例外・ログの粒度調整（FITS 読み込み失敗、ヘッダ検査など）。
- 並列化の検討（I/O バウンド: `concurrent.futures.ThreadPoolExecutor`）。