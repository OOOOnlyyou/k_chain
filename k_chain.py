import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mplfinance as mpf
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class KChain(object):
    def __init__(self, chanK):
        self.__rawChanK = chanK
        self.__chanK = self.__toChanK(chanK)
        self.__kState = list()  # k线状态->五种状态：(0,0)、(1,1)、(-1,1)、(1,0)、(-1,0)
        self.__fenTypes = list()  # 分型类型列表 1,-1构成
        self.__fenIdx = list()  # 分型对应k的下标
        self.__biIdx = list()  # 笔对应的k线下标
        self.__biFenTypes = list()  # 笔分型类型列表 1,-1构成
        self.__frsBiType = 0  # 起始笔的走势，biIdx 奇数下标就是相反走势 1、向上，-1向下
        self.__segIdx = list()  # 线段对应的k线下标
        self.__segFenTypes = list()
        self.__pivotSet = list()  # 存放中枢的信息集:[(leftBound,rightBound, maxLow, minHigh),...]

    def __kLineState(self, chanK):
        for i in range(chanK.shape[0]):
            if not self.__kState:
                self.__kState.append((0, 0))
                continue
            if len(self.__kState) == 1:
                if chanK['high'][i] > chanK['high'][i - 1]:
                    self.__kState.append((1, 1))
                else:
                    self.__kState.append((-1, 0))
                continue
            if chanK['high'][i] > chanK['high'][i - 1]:
                if self.__kState[-1] in [(1, 1), (-1, 0)]:
                    self.__kState.append((1, 1))
                else:
                    self.__kState.append((-1, 0))
            else:
                if self.__kState[-1] in [(-1, 1), (1, 0)]:
                    self.__kState.append((-1, 1))
                else:
                    self.__kState.append((1, 0))

    # 解析包含关系
    @staticmethod
    def __mergeKLine(raw_chanK, in_chan=False):
        _newColumns = ['endDate'] + list(raw_chanK.keys()) if not in_chan else list(raw_chanK.keys())
        _newChanK = pd.DataFrame(
            np.zeros((raw_chanK.shape[0], len(_newColumns))), index=raw_chanK.index,
            columns=_newColumns) if not in_chan else raw_chanK

        def setNewValue(dt, value):
            _newChanK.loc[dt, _newColumns] = value
            return dt

        # 比较上一个(n-1)包含的时间点
        _last1K = 0
        # 比较n-2包含的时间点
        _last2K = 0
        for i, dt in enumerate(raw_chanK.index):
            if i == 0:
                _last1K = setNewValue(dt, [dt] + list(raw_chanK.values[i])) if not in_chan else dt
                continue
            if in_chan and (raw_chanK['endDate'][dt] == 0 or i == 0): continue

            # 非包含情况
            if (raw_chanK['high'][dt] > _newChanK['high'][_last1K] and raw_chanK['low'][dt] > _newChanK['low'][_last1K]) \
                    or (raw_chanK['high'][dt] < _newChanK['high'][_last1K] and raw_chanK['low'][dt] < _newChanK['low'][
                _last1K]):
                _last2K = _last1K
                if in_chan:
                    _last1K = dt
                    continue
                _last1K = setNewValue(dt, [dt] + list(raw_chanK.values[i]))
                continue

            # 包含情况
            endDate = dt if not in_chan else _newChanK['endDate'][dt]
            if raw_chanK['high'][dt] >= _newChanK['high'][_last1K] and raw_chanK['low'][dt] <= _newChanK['low'][
                _last1K]:
                new_high, new_low = 0, 0
                # 向上处理
                if _newChanK['high'][_last1K] > _newChanK['high'][_last2K]:
                    new_high, new_low = raw_chanK['high'][dt], _newChanK['low'][_last1K]
                # 向下处理
                else:
                    new_high, new_low = _newChanK['high'][_last1K], raw_chanK['low'][dt]
                _last1K = setNewValue(_last1K,
                                      [endDate, _newChanK['open'][_last1K],
                                       new_high,
                                       new_low,
                                       raw_chanK['close'][dt],
                                       raw_chanK['vol'][dt] + _newChanK['vol'][_last1K]])
            if raw_chanK['high'][dt] < _newChanK['high'][_last1K] and raw_chanK['low'][dt] > _newChanK['low'][_last1K]:
                new_high, new_low = 0, 0
                # 向上处理
                if _newChanK['high'][_last1K] > _newChanK['high'][_last2K]:
                    new_high, new_low = _newChanK['high'][_last1K], raw_chanK['low'][dt]
                # 向下处理
                else:
                    new_high, new_low = raw_chanK['high'][dt], _newChanK['low'][_last1K]
                _last1K = setNewValue(_last1K,
                                      [endDate, _newChanK['open'][_last1K],
                                       new_high,
                                       new_low,
                                       raw_chanK['close'][dt],
                                       raw_chanK['vol'][dt] + _newChanK['vol'][_last1K]])
            if in_chan:
                _newChanK.at[dt, 'endDate'] = 0
        _newChanK = _newChanK[_newChanK['endDate'] != 0]
        return _newChanK

    # 处理k线成缠论k线
    @staticmethod
    def __toChanK(raw_chanK):
        chank = KChain.__mergeKLine(raw_chanK)
        while True:
            chank_ = KChain.__mergeKLine(chank, in_chan=True)
            if len(chank.index) == len(chank_.index):
                break
            chank = chank_
        return chank

    # 过滤连续同类分型
    # 如果 连续顶顶，或底底： 顶：high最大的顶， 低：low最小的低
    def __filterFenType(self, fidx: int, ft: int):
        if len(self.__fenIdx) == 0:
            self.__fenIdx.append(fidx)
            self.__fenTypes.append(ft)
            return
        fenType_bf = self.__fenTypes[len(self.__fenTypes) - 1]
        if fenType_bf == ft:
            fenType_bf, fenIdx_bf = self.__fenTypes.pop(), self.__fenIdx.pop()
            fidx = fenIdx_bf if (ft == 1 and self.__chanK['high'][fenIdx_bf] > self.__chanK['high'][fidx]) or (
                    ft == -1 and self.__chanK['low'][fenIdx_bf] < self.__chanK['low'][fidx]) else fidx
        self.__fenIdx.append(fidx)
        self.__fenTypes.append(ft)

    # 找顶底分型
    # 顶：1 低：-1
    def __findFenType(self):
        for i, date in enumerate(self.__chanK.index):
            if i == 0 or i == len(self.__chanK.index) - 1: continue
            # 顶分型
            if (self.__chanK['high'][i + 1] < self.__chanK['high'][i] > self.__chanK['high'][i - 1]) and \
                    (self.__chanK['low'][i + 1] < self.__chanK['low'][i] > self.__chanK['low'][i - 1]):
                self.__filterFenType(i, 1)
            # 底分型
            if (self.__chanK['high'][i + 1] > self.__chanK['high'][i] < self.__chanK['high'][i - 1]) and \
                    (self.__chanK['low'][i + 1] > self.__chanK['low'][i] < self.__chanK['low'][i - 1]):
                self.__filterFenType(i, -1)

    # 判断分型破坏
    def __judgeFenBreak(self, fenType, bf, af):
        return (fenType == -1 and self.__chanK['low'][af] < self.__chanK['low'][bf]) or (
                fenType == 1 and self.__chanK['high'][af] > self.__chanK['high'][bf])

    # 判断笔破坏
    def __judgeBiBreak(self, idxb, idxa):
        _fenType0 = self.__fenTypes[idxb]
        _fenType1 = -_fenType0
        _break = False
        _breaki_k, _breakj_k = 0, 0
        for k in range(idxb, idxa)[2::2]:
            # 当前i分型破坏
            if self.__judgeFenBreak(_fenType0, self.__fenIdx[idxb], self.__fenIdx[k]):
                _break, _breaki_k = True, k
                break
        for k in range(idxb, idxa)[1::2]:
            # 末尾j分型破坏
            if self.__judgeFenBreak(_fenType1, self.__fenIdx[idxa], self.__fenIdx[k]):
                _break, _breakj_k = True, k
                break
        return _break, _breaki_k, _breakj_k

    def __reAssignBi(self, breakBi, breakBj, i, j):
        _toConBfIdx = i + 1 if len(self.__biIdx) == 0 else breakBi
        if breakBi > 0 and len(self.__biIdx) > 0:
            fb_ = self.__biIdx.pop()
            self.__biFenTypes.pop()
            self.__biIdx.append(self.__fenIdx[breakBi])
            self.__biFenTypes.append(self.__fenTypes[breakBi])
            if len(self.__biIdx) > 1 and 0 < breakBj < breakBi and self.__judgeFenBreak(self.__fenTypes[i - 1],
                                                                                        self.__biIdx[
                                                                                            len(self.__biIdx) - 2],
                                                                                        self.__fenIdx[breakBj]):
                fa_ = self.__biIdx.pop()
                self.__biFenTypes.pop()
                fb_ = self.__biIdx.pop()
                self.__biFenTypes.pop()
                self.__biIdx.append(self.__fenIdx[breakBj])
                self.__biFenTypes.append(self.__fenTypes[breakBj])
                _toConBfIdx = breakBj
            return _toConBfIdx, -1  # -1：break 1:continue 0:不执行

        if breakBj > 0:
            if j + 2 >= len(self.__fenIdx) and len(self.__biIdx) > 1 and self.__judgeFenBreak(self.__fenTypes[i - 1],
                                                                                              self.__biIdx[
                                                                                                  len(self.__biIdx) - 2],
                                                                                              self.__fenIdx[breakBj]):
                fa_ = self.__biIdx.pop()
                self.__biFenTypes.pop()
                fb_ = self.__biIdx.pop()
                self.__biFenTypes.pop()
                self.__biIdx.append(self.__fenIdx[breakBj])
                self.__biFenTypes.append(self.__fenTypes[breakBj])
                _toConBfIdx = breakBj
                return _toConBfIdx, -1  # -1：break 1:continue 0:不执行
            return _toConBfIdx, 1  # -1：break 1:continue 0:不执行
        return _toConBfIdx, 0  # -1：break 1:continue 0:不执行

    # 分型构成笔
    # 构成笔条件，1、顶低分型间隔了n个k线， 2、中间不会出现比第一个分型结构更高（顶）或更低（底）的分型，否则线段破坏，连接上一笔
    def __getBi(self):
        _least_khl_num = 3  # 分笔间隔的最小 chanK 数量 中间排除顶低的chanK
        _toConBfIdx = 0  # 连接到上一笔末尾的 分型idx

        for i, kIdx in enumerate(self.__fenIdx):
            if i < _toConBfIdx or i == len(self.__fenIdx) - 1: continue
            # 后面没有符合条件的笔
            if len(self.__biIdx) > 1 and _toConBfIdx == 0: break
            _toConBfIdx = 0
            for j in range(len(self.__fenIdx))[i + 1::2]:
                if (self.__fenIdx[j] - kIdx) > _least_khl_num:
                    # breakType True 同分型， False 末尾分型
                    flag, breakBi, breakBj = self.__judgeBiBreak(i, j)
                    if flag:
                        _toConBfIdx, _bcn = self.__reAssignBi(breakBi, breakBj, i, j)
                        if _bcn == -1: break
                        if _bcn == 1: continue
                    if len(self.__biIdx) == 0:
                        self.__biIdx.append(kIdx)
                        self.__biFenTypes.append(self.__fenTypes[i])
                        self.__frsBiType = -self.__fenTypes[i]
                        self.__biIdx.append(self.__fenIdx[j])
                        self.__biFenTypes.append(self.__fenTypes[j])
                        _toConBfIdx, _bcn = self.__reAssignBi(breakBi, breakBj, i, j)
                        if _bcn == -1 or _bcn == 1:
                            self.__biIdx = []
                            self.__biFenTypes = []
                            _toConBfIdx = i + 1
                            break
                        _toConBfIdx = j
                        break
                    self.__biIdx.append(self.__fenIdx[j])
                    self.__biFenTypes.append(self.__fenTypes[j])
                    _toConBfIdx = j
                    break

    # 判断线段破坏
    def __judgeSegBreak(self, fenType, afPrice, idx):
        return (fenType == -1 and afPrice < self.__chanK['low'][idx]) or (
                fenType == 1 and afPrice > self.__chanK['high'][idx])

    # 重构线段
    def __rebuildSeg(self, txIdx, nxIdx):
        _xdIdxn = self.__segIdx
        _xfenTypesn = self.__segFenTypes
        if len(_xdIdxn) == 0 or txIdx == -1 or nxIdx == -1: return 0, _xdIdxn, _xfenTypesn
        for m in range(-len(_xdIdxn) + 1, 1)[1::2]:
            k = -m
            # 满足逆向不破坏
            if (_xfenTypesn[k] == -1 and self.__chanK['low'][_xdIdxn[k]] < self.__chanK['low'][nxIdx]) \
                    or (_xfenTypesn[k] == 1 and self.__chanK['high'][_xdIdxn[k]] > self.__chanK['high'][nxIdx]):
                for n in range(-len(_xdIdxn) + 1, m):
                    _xfenTypesn.pop()
                    _xdIdxn.pop()
                _xdIdxn.append(txIdx)
                _xfenTypesn.append(-_xfenTypesn[len(self.__segFenTypes) - 1])
                return self.__biIdx.index(txIdx), _xdIdxn, _xfenTypesn
        _xdIdxn = []
        _xfenTypesn = []
        return self.__biIdx.index(nxIdx), _xdIdxn, _xfenTypesn

    # 最终线段生成
    # 1、遍历相对高低点，判断线段破坏，
    #  破坏以后,如果总长度是0，i可以后移，否则重构之前的线段
    #  重构规则，找破坏点相对高点/低点，如果存在线段高点/低点>/<破坏高点/低点， 则连接此线段
    # 2、形成线段中间至少有两笔点j-i>2
    def __getSegment(self):
        _lenBiIdx = len(self.__biIdx)
        if _lenBiIdx == 0: return self.__segIdx, self.__segFenTypes

        afIdx = 0
        for i, idx in enumerate(self.__biIdx):
            if afIdx < 0: break  # 线段破坏以后没有合适线段
            fenType = 1 if (self.__frsBiType == 1 and i % 2 == 1) or (self.__frsBiType == -1 and i % 2 == 0) else -1
            # 符合要求的连段
            if i < afIdx: continue
            # 找同向相对高低点
            afPrice = 0 if fenType == -1 else 10000
            tongxiang_price_ = 10000 - afPrice
            nixiang_idx, tongxiang_idx = -1, -1
            i_continued = False
            for j in range(i + 1, _lenBiIdx, 2):
                # 同向相对高低点
                if (fenType == -1 and tongxiang_price_ > self.__chanK['low'][self.__biIdx[j - 1]]) \
                        or (fenType == 1 and tongxiang_price_ < self.__chanK['high'][self.__biIdx[j - 1]]):
                    tongxiang_idx = self.__biIdx[j - 1]
                    tongxiang_price_ = self.__chanK['high'][tongxiang_idx] if fenType == 1 else self.__chanK['low'][
                        tongxiang_idx]

                # 线段破坏
                # 同向破坏
                if self.__judgeSegBreak(fenType, tongxiang_price_, idx) and idx != tongxiang_idx:
                    #                 print '同向已经破坏'
                    afIdx, self.__segIdx, self.__segFenTypes = self.__rebuildSeg(tongxiang_idx, nixiang_idx)
                    i_continued = True
                    break
                if (fenType == -1 and self.__chanK['high'][self.__biIdx[j]] > afPrice) or (
                        fenType == 1 and self.__chanK['low'][self.__biIdx[j]] < afPrice):
                    nixiang_idx = self.__biIdx[j]
                    afPrice = self.__chanK['high'][nixiang_idx] if fenType == -1 else self.__chanK['low'][nixiang_idx]

                    # 线段不符合要求
                    if j - i <= 2: continue
                    # 逆向破坏
                    if len(self.__segIdx) == 0:
                        if self.__judgeSegBreak(fenType, tongxiang_price_, idx):
                            i_continued = True
                            break
                        self.__segFenTypes.append(fenType)
                        self.__segIdx.append(idx)
                        self.__segIdx.append(self.__biIdx[j])
                        self.__segFenTypes.append(-fenType)
                    else:
                        # 不用同向线段连接
                        self.__segFenTypes.append(-self.__segFenTypes[len(self.__segFenTypes) - 1])
                        self.__segIdx.append(self.__biIdx[j])
                    afIdx = j
                    i_continued = True
                    break
                else:
                    continue
            if not i_continued and len(self.__segIdx) > 0:
                # 都不符合要求时，最后重构最小线段
                last_idx = self.__segIdx.pop()
                last_type = self.__segFenTypes[len(self.__segFenTypes) - 1]
                for j in range(self.__biIdx.index(last_idx), len(self.__biIdx))[2::2]:
                    if self.__judgeSegBreak(last_type, self.__chanK['low'][self.__biIdx[j]] if last_type == -1 else
                    self.__chanK['high'][self.__biIdx[j]],
                                            last_idx):
                        last_idx = self.__biIdx[j]
                self.__segIdx.append(last_idx)
                break

    def __getPivot(self, biPrices: list):
        # 计算中枢
        # 注意：一个中枢至少有三笔
        _biPointNum = len(self.__biIdx)
        _leftBound, _rightBound = 0, 0
        _maxLow, _minHigh = 0, 0
        # 不满足基本条件
        if _biPointNum < 5: return
        _cover = 0  # 记录走势的重叠区，至少为3才能画中枢
        i = 2
        _rightIdx = 0
        while i < _biPointNum:
            if _cover == 0:
                _rightBound = 0
                _leftBound = self.__biIdx[i - 1]
                # 所观察分型的上一笔是往上的一笔
                if biPrices[i] >= biPrices[i - 1]:
                    _minHigh = biPrices[i]
                    _maxLow = biPrices[i - 1]
                # 所观察分型的上一笔是往下的一笔
                else:
                    _maxLow = biPrices[i]
                    _minHigh = biPrices[i - 1]
                _cover += 1
                i += 1
                continue

            # 往上的一笔
            if biPrices[i] >= biPrices[i - 1]:
                # 已经没有重叠区域了
                if biPrices[i] < _maxLow:
                    # 判断是否满足中枢定义
                    if _cover >= 3:
                        _rightBound = self.__biIdx[_rightIdx]
                        self.__pivotSet.append((_leftBound, _rightBound, _maxLow, _minHigh))
                    # 该中枢结束，为下一个中枢初始准备
                    if _cover > 3:
                        i = _rightIdx + 1
                    # if _cover == 2:
                    #     i -= 1
                    _cover = 0
                else:
                    _cover += 1
                    # 有重叠区域
                    # 计算更窄的中枢价格区间
                    if _cover <= 3:
                        _minHigh = min(biPrices[i], _minHigh)
                        _maxLow = max(biPrices[i - 1], _maxLow)
                        if _cover == 3:
                            _rightIdx = i
                    elif _maxLow <= biPrices[i] <= _minHigh:
                        _rightIdx = i
                    i += 1
            # 往下的一笔
            elif biPrices[i] < biPrices[i - 1]:
                # 已经没有重叠区域了
                if biPrices[i] > _minHigh:
                    # 判断是否满足中枢定义
                    if _cover >= 3:
                        _rightBound = self.__biIdx[_rightIdx]
                        self.__pivotSet.append((_leftBound, _rightBound, _maxLow, _minHigh))
                    if _cover > 3:
                        i = _rightIdx + 1
                    # if _cover == 2:
                    #     i -= 1
                    _cover = 0
                else:
                    _cover += 1
                    # 有重叠区域
                    # 计算更窄的中枢价格区间
                    if _cover <= 3:
                        _minHigh = min(biPrices[i - 1], _minHigh)
                        _maxLow = max(biPrices[i], _maxLow)
                        if _cover == 3:
                            _rightIdx = i
                    elif _maxLow <= biPrices[i] <= _minHigh:
                        _rightIdx = i
                    i += 1
        if _cover >= 3:
            _rightBound = self.__biIdx[_rightIdx]
            self.__pivotSet.append((_leftBound, _rightBound, _maxLow, _minHigh))

    # 计算坐标
    def __getXY(self, idxSeq: list, fenTypeSeq: list):
        _X, _Y = [], []
        for i in range(len(idxSeq)):
            if idxSeq[i]:
                _fenType = fenTypeSeq[i]
                if _fenType == 1:
                    _X.append(self.__chanK.index[idxSeq[i]])
                    _Y.append(self.__chanK['high'][idxSeq[i]])
                if _fenType == -1:
                    _X.append(self.__chanK.index[idxSeq[i]])
                    _Y.append(self.__chanK['low'][idxSeq[i]])
        return _X, _Y

    def __plotPivot(self, ax, pivotSet):
        # 绘制中枢
        if not pivotSet: return
        for s, e, l, h in pivotSet:
            start_point = (s, l)
            width = e - s
            height = h - l
            linestyle = '--' if e == self.__biIdx[-1] else '-'
            ax.add_patch(
                patches.Rectangle(
                    start_point,  # (x,y)
                    width,  # width
                    height,  # height
                    linewidth=2,
                    edgecolor='m',
                    facecolor='none',
                    linestyle=linestyle
                )
            )
            ax.hlines(xmin=s, xmax=e, y=(l + h) / 2, color='m', lw=2, linestyle='--')

    def plot(self, lines, pivotSet, mav=(5, 10, 20, 30)):
        selfColor = mpf.make_marketcolors(up='r',
                                          down='g',
                                          edge='inherit',
                                          wick='inherit',
                                          volume='inherit')
        # 设置图表的背景色
        selfStyle = mpf.make_mpf_style(marketcolors=selfColor,
                                       figcolor='(0.82, 0.83, 0.85)',
                                       gridcolor='(0.82, 0.83, 0.85)')
        fig, ax = plt.subplots(figsize=(40, 20))
        mpf.plot(self.__chanK,
                 type='candle',
                 style=selfStyle,
                 alines=dict(
                     alines=lines,
                     colors=['y', 'b'],
                     linestyle=['-', '-'],
                     linewidths=[1, 3],
                 ),
                 datetime_format='%Y-%m-%d',
                 # figsize=(40,20),
                 scale_width_adjustment=dict(candle=2.5, lines=1),
                 xrotation=15,
                 mav=mav,
                 ax=ax
                 )
        self.__plotPivot(ax, pivotSet)
        plt.show()

    def plotKState(self, rawK=False):
        _chanK = self.__rawChanK if rawK else self.__chanK
        self.__kLineState(_chanK)
        selfColor = mpf.make_marketcolors(up='r',
                                          down='g',
                                          edge='inherit',
                                          wick='inherit',
                                          volume='inherit')
        # 设置图表的背景色
        selfStyle = mpf.make_mpf_style(marketcolors=selfColor,
                                       figcolor='(0.82, 0.83, 0.85)',
                                       gridcolor='(0.82, 0.83, 0.85)')
        fig, ax = plt.subplots(figsize=(40, 20))
        mpf.plot(_chanK,
                 type='candle',
                 style=selfStyle,
                 datetime_format='%Y-%m-%d',
                 scale_width_adjustment=dict(candle=2.5, lines=1),
                 xrotation=15,
                 mav=(5, 10, 20, 30),
                 ax=ax
                 )
        for x, state in zip(range(_chanK.shape[0]), self.__kState):
            y = _chanK.loc[_chanK.index[x], 'high']
            ax.text(x, y, state, fontweight='bold')
        plt.show()

    def main(self):
        self.__findFenType()
        self.__getBi()
        self.__getSegment()
        _biX, _biY = self.__getXY(self.__biIdx, self.__biFenTypes)
        _segX, _segY = self.__getXY(self.__segIdx, self.__segFenTypes)
        lines = [list(zip(_biX, _biY)), list(zip(_segX, _segY))]
        self.__getPivot(_biY)
        self.plot(lines, self.__pivotSet)
