"""
定义算法类
"""
from abc import ABC, abstractmethod
from dataClass import *
from typing import Dict, Any, Tuple, List


class AlgorithmBaseClass(ABC):
    """算法类的基类"""
    def __init__(self) -> None:
        self.riskProfile: np.ndarray| None = None  # 不同扇区风险概况，扇区的起点指向正北方向，顺时针为正

    @abstractmethod
    def compute_risk_profile(self) -> Any:
        """统计每个扇区的风险值"""
        pass
    
    @abstractmethod
    def select_action(self) -> Any:
        """选择动作"""
        pass


class ExampleAlgorithm(AlgorithmBaseClass):
    """示例算法类"""
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

    def select_action(self, ownState: StateEstimate, trackStates: Dict[str, StateEstimate]) -> BlendedAction:
        """
        示例算法选择动作，综合所有入侵者，给出环绕一圈的风险评估，选择水平动作
        :param ownState: 无人机自身状态
        :param trackStates: 所有入侵者的状态
        :return: 融合动作
        """
        # 计算每个入侵者的TRM状态
        trmStates = []
        for _, targetState in trackStates.items():
            trmStates.append(self._calculate_trm_state(ownState, targetState))
        # 统计无人机周围一圈的风险分布，分成N个扇区，扇区的起点指向正北方向，顺时针为正
        self.riskProfile, sectorIntruders = self.compute_risk_profile(trmStates, self.cfg.sectorNum)
        # 根据风险分布选择水平动作
        horizontalAction = self._lookup_horizontal_action(self.riskProfile, sectorIntruders)
        # 根据风险分布选择垂直动作
        verticalAction = self._lookup_vertical_action(trmStates)
        # 综合水平动作和垂直动作
        blendedAction = BlendedAction(horizontalAction, verticalAction)
        return blendedAction

    def _calculate_trm_state(self, ownState: StateEstimate, trackState: StateEstimate) -> TRMState:
        """
        计算TRM状态变量
        :param ownState: 无人机自身状态
        :param trackState: 入侵者状态
        :return: TRM状态变量
        """
        # 计算相对位置和速度
        relativePosition = Position(
            east=trackState.position.east - ownState.position.east,
            north=trackState.position.north - ownState.position.north,
            up=trackState.position.up - ownState.position.up
        )
        relativeVelocity = Velocity(
            eastVelocity=trackState.velocity.eastVelocity - ownState.velocity.eastVelocity,
            northVelocity=trackState.velocity.northVelocity - ownState.velocity.northVelocity,
            upVelocity=trackState.velocity.upVelocity - ownState.velocity.upVelocity
        )
        # 计算水平距离
        horizontalRange = np.sqrt(relativePosition.east ** 2 + relativePosition.north ** 2)
        relativeHorizontalPosition = np.array([relativePosition.east, relativePosition.north])
        # 计算相对航迹角 (ψ)以自己航迹角为参考，正为靠近，负为远离
        ownHeading = np.arctan2(ownState.velocity.eastVelocity, ownState.velocity.northVelocity)
        relativeTrackAngle = np.arctan2(trackState.velocity.eastVelocity, trackState.velocity.northVelocity) - ownHeading
        relativeTrackAngle = (relativeTrackAngle + np.pi) % (2 * np.pi) - np.pi  # 归一化角度到[-π, π]
        # 计算相对方位角 (θ)以自己方位角为参考，正为右边，负为左边
        ownHeading = np.arctan2(ownState.position.east, ownState.position.north)
        relativeBearing = np.arctan2(trackState.position.east, trackState.position.north) - ownHeading
        relativeBearing = (relativeBearing + np.pi) % (2 * np.pi) - np.pi  # 归一化角度到[-π, π]
        # 计算垂直分离即将丢失时间 (τv)
        if abs(relativeVelocity.upVelocity) > 0.001:  # 避免除以零
            timeToVerticalLoss = abs(relativePosition.up / relativeVelocity.upVelocity)
        else:
            timeToVerticalLoss = float('inf')   
        # 计算水平分离即将丢失时间 (τ)
        relativePositionVeclocity = np.array([relativePosition.east, relativePosition.north])  # 水平相对位置
        relativeVelocityVeclocity = np.array([relativeVelocity.eastVelocity, relativeVelocity.northVelocity])  # 水平相对速度
        horizontalRange = np.linalg.norm(relativePositionVeclocity)  # 水平距离
        if horizontalRange > 1e-6:
            direction = relativePositionVeclocity / horizontalRange  # 单位方向向量
            closingSpeed = np.dot(relativeVelocityVeclocity, direction)  # 投影速度
            if abs(closingSpeed) > 1e-3:
                timeToHorizontalLoss = horizontalRange / -closingSpeed  # 加负号是为了，正为靠近，负为远离
            else:
                timeToHorizontalLoss = float('inf')
        else:
            timeToHorizontalLoss = 0.0
        return TRMState(
            horizontalRange=horizontalRange,
            relativeTrackAngle=relativeTrackAngle,
            relativeBearing=relativeBearing,
            relativeHorizontalPosition=relativeHorizontalPosition,
            ownHorizontalSpeed=np.sqrt(ownState.velocity.eastVelocity**2 + ownState.velocity.northVelocity**2),
            intruderHorizontalSpeed=np.sqrt(trackState.velocity.eastVelocity**2 + trackState.velocity.northVelocity**2),
            timeToVerticalLoss=timeToVerticalLoss,
            relativeAltitude=relativePosition.up,
            ownVerticalSpeed=ownState.velocity.upVelocity,
            intruderVerticalSpeed=trackState.velocity.upVelocity,
            timeToHorizontalLoss=timeToHorizontalLoss
        )
    
    def compute_risk_profile(self, trmStates: list[TRMState], sectorNum: int = 3) -> Tuple[np.ndarray, List[List[TRMState]]]:
        """
        统计每个扇区的风险值和扇区内入侵者（按距离排序，近的在前）
        :param trmStates: TRM状态列表
        :param sectorNum: 扇区数量
        :return: (riskProfile, sectorIntruders)
        riskProfile: np.ndarray, shape=(sectorNum,)
        sectorIntruders: List[List[TRMState]], 每个扇区一个列表，按距离升序
        """
        # 初始化sectorNum个扇区的风险概况
        riskProfile = np.zeros(sectorNum)
        sectorIntruders = [[] for _ in range(sectorNum)]
        sectorAngle = 2 * np.pi / sectorNum  # 每个扇区对应的角度
        for state in trmStates:
            angle = np.arctan2(state.relativeHorizontalPosition[0], state.relativeHorizontalPosition[1]) % (2 * np.pi)  # 入侵者方位角，归一化到[0, 2π]
            idx = int(angle // sectorAngle) # 计算扇区索引
            risk = 1.0 / (state.horizontalRange + 1e-3)
            riskProfile[idx] += risk
            sectorIntruders[idx].append(state)
        # 对每个扇区的入侵者按距离排序
        for intruders in sectorIntruders:
            intruders.sort(key=lambda s: s.horizontalRange)
        return riskProfile, sectorIntruders

    def _lookup_horizontal_action(self, riskProfile: np.ndarray, sectorIntruders: List[List[TRMState]]) -> HorizontalAction:
        """
        查询水平策略表，综合相对方位角、接近时间、相对航迹角
        :param riskProfile: 风险概况
        :param sectorIntruders: 每个扇区的入侵者列表
        :return: 水平动作
        """
        max_idx = int(np.argmax(riskProfile))
        if not sectorIntruders[max_idx]:
            return HorizontalAction.NONE
        # 取该扇区最近的入侵者
        state = sectorIntruders[max_idx][0]
        # 紧急威胁：接近时间很短
        if state.timeToHorizontalLoss < 10:
            if state.relativeBearing < 0:
                #  目标在左侧同时向靠近自己方向移动
                if state.relativeTrackAngle > 0:
                    return HorizontalAction.TURN_RIGHT_FAST
                # 目标在左侧同时向远离自己方向移动
                else:
                    return HorizontalAction.TURN_RIGHT_SLOW
            else:
                # 目标在右侧同时向靠近自己方向移动
                if state.relativeTrackAngle < 0:
                    return HorizontalAction.TURN_LEFT_FAST
                # 目标在右侧同时向远离自己方向移动
                else:
                    return HorizontalAction.TURN_LEFT_SLOW
        # 中等威胁
        elif state.timeToHorizontalLoss < 30:
            if state.relativeBearing < 0:
                # 目标在左侧
                return HorizontalAction.TURN_RIGHT_SLOW
            else:
                # 目标在右侧
                return HorizontalAction.TURN_LEFT_SLOW
        # 低威胁或无威胁
        else:
            return HorizontalAction.NONE

    def _lookup_vertical_action(self, trmStates: list[TRMState]) -> VerticalAction:
        """
        查询垂直策略表，综合相对高度、接近时间
        :param trmStates: TRM状态列表
        :return: 垂直动作
        """
        # 找到timeToVerticalLoss最小且小于阈值的入侵者
        minTime = float('inf')
        threatState = None
        for state in trmStates:
            if state.timeToVerticalLoss < 20 and state.timeToVerticalLoss < minTime:
                minTime = state.timeToVerticalLoss
                threatState = state
        if threatState is not None:
            if threatState.relativeAltitude > 0:
                return VerticalAction.DESCEND
            elif threatState.relativeAltitude < 0:
                return VerticalAction.CLIMB
            else:
                # 高度完全一致时，主动选择上升
                return VerticalAction.CLIMB
        # 新增：如果没有即将丢失分离的，但有高度差或高度一致且双方垂直速度都很小
        for state in trmStates:
            if abs(state.relativeAltitude) <= 1.0 and \
            abs(state.ownVerticalSpeed) < 0.1 and abs(state.intruderVerticalSpeed) < 0.1:
                # 高度一致，主动选择上升
                return VerticalAction.CLIMB
            elif abs(state.relativeAltitude) > 1.0 and \
                abs(state.ownVerticalSpeed) < 0.1 and abs(state.intruderVerticalSpeed) < 0.1:
                # 有高度差，往远离对方的方向动作
                if state.relativeAltitude > 0:
                    return VerticalAction.DESCEND
                else:
                    return VerticalAction.CLIMB
        return VerticalAction.NONE

    def get_risk_profile(self) -> np.ndarray:
        """
        获取风险概况
        :return: 风险概况
        """
        return self.riskProfile


if __name__ == "__main__":
    pass