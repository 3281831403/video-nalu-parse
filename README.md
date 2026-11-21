# video-nalu-parse

## 题面

做码流分析时，SPS 解出来的分辨率总比实际宽、切片还丢帧：起始码只认 4 字节形式，3 字节起始码的 NAL 整段漏切；去掉起始码时多切 1 字节，NAL 头丢失；nal_unit_type 取了高 5 位，类型全部错位；去防竞争字节只删 03 而不恢复数据，RBSP 被截断；插入防竞争字节时对 00 00 00 连续插 03，生成不合法的转义序列；SPS 宽高不按 frame_cropping 修正，画面右侧和底部多出黑边；PPS 的 pic_parameter_set_id 用定长读取而不是 ue(v)，id 解析错误；空 NAL 不报错，后续读取头部时越界；批量解析共用同一个缓冲，不同流之间串数据；nal_unit_type 为 0 时照常解析，进入未定义行为。nalu/bitstream.py 是核心，tests/test_bitstream.py 15个红。要修的是：split_nals 同时识别 3 字节和 4 字节起始码并完整保留 NAL 头，nal_type 取低 5 位且对类型 0 抛 NaluError，strip_emulation 正确还原 00 00 序列，insert_emulation 只在必要位置插入一个 03 并与 strip_emulation 往返一致，parse_sps 解析 profile/level 并按 frame_cropping 修正宽高，parse_pps 用 ue(v) 读取 id 与熵编码标志，validate_stream 报告空 NAL、禁止类型和长度越界，parse_batch 隔离每条流的缓冲并在任一流非法时整体回滚。

## 运行方式

```bash
cd D:\gsb-20261119\video-nalu-parse
python -m pytest tests -q
```

初始状态应恰好输出：

```text
15 failed
```

15 条测试全部因桩函数 `raise NotImplementedError` 而失败；收集、导入和语法检查必须通过，不能出现 `error`。实现后测试应全部通过。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 测试约束

`tests/` 是验收测试，不可修改、删除或新增；修复只能落在 `nalu/bitstream.py`。所有码流都用 `bytes` 构造，实现只使用 Python 标准库（`struct`）和 pytest，离线运行、无随机、不联网。NAL 字节序列比较允许 1 字节的 `trailing_zero_8bits` 容差，但 NAL 头、起始码和插入的 `03` 必须精确匹配。
