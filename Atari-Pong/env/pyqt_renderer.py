from __future__ import annotations

import time
from typing import Any

from env.pong_env import PongEnv, PongSnapshot

try:
    from PyQt6.QtCore import QEvent, QTimer, Qt
    from PyQt6.QtGui import QColor, QFont, QPainter, QPen
    from PyQt6.QtWidgets import QApplication, QWidget

    PYQT6_AVAILABLE = True
except ModuleNotFoundError:
    QTimer = Any  # type: ignore[assignment]
    QEvent = Any  # type: ignore[assignment]
    Qt = Any  # type: ignore[assignment]
    QColor = Any  # type: ignore[assignment]
    QFont = Any  # type: ignore[assignment]
    QPainter = Any  # type: ignore[assignment]
    QPen = Any  # type: ignore[assignment]
    QApplication = Any  # type: ignore[assignment]
    QWidget = object  # type: ignore[assignment]
    PYQT6_AVAILABLE = False


def _require_pyqt6() -> None:
    if not PYQT6_AVAILABLE:
        raise ImportError("PyQt6 is required for rendering. Install with `pip install PyQt6`.")


def _qt_app() -> QApplication:
    _require_pyqt6()
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class PongCanvas(QWidget):
    def __init__(self, title: str = "Atari-Pong", width_px: int = 960, height_px: int = 540) -> None:
        _require_pyqt6()
        _qt_app()
        super().__init__()
        self.setWindowTitle(title)
        self.resize(width_px, height_px)
        self._snapshot: PongSnapshot | None = None

    def set_snapshot(self, snapshot: PongSnapshot) -> None:
        self._snapshot = snapshot
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        if self._snapshot is None:
            return

        s = self._snapshot
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        field_w = rect.width()
        field_h = rect.height()

        painter.fillRect(rect, QColor(12, 18, 24))

        line_pen = QPen(QColor(220, 220, 220, 160), 2)
        painter.setPen(line_pen)
        painter.drawLine(field_w // 2, 0, field_w // 2, field_h)

        paddle_color = QColor(240, 240, 240)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(paddle_color)

        paddle_px_h = int(s.paddle_h / s.height * field_h)
        paddle_px_w = max(6, int(s.paddle_w / s.width * field_w))

        left_px_x = int(s.left_paddle_x / s.width * field_w)
        right_px_x = int(s.right_paddle_x / s.width * field_w)
        left_px_y = int(s.left_paddle_y / s.height * field_h)
        right_px_y = int(s.right_paddle_y / s.height * field_h)

        painter.drawRoundedRect(
            left_px_x - paddle_px_w // 2,
            left_px_y - paddle_px_h // 2,
            paddle_px_w,
            paddle_px_h,
            4,
            4,
        )
        painter.drawRoundedRect(
            right_px_x - paddle_px_w // 2,
            right_px_y - paddle_px_h // 2,
            paddle_px_w,
            paddle_px_h,
            4,
            4,
        )

        ball_px_x = int(s.ball_x / s.width * field_w)
        ball_px_y = int(s.ball_y / s.height * field_h)
        ball_px_r = max(5, int(s.ball_r / s.width * field_w))
        painter.setBrush(QColor(121, 255, 169))
        painter.drawEllipse(ball_px_x - ball_px_r, ball_px_y - ball_px_r, ball_px_r * 2, ball_px_r * 2)

        painter.setPen(QColor(250, 250, 250))
        painter.setFont(QFont("Consolas", 20, weight=QFont.Weight.Bold))
        score_text = f"{s.score_left:02d}  :  {s.score_right:02d}"
        painter.drawText(0, 12, field_w, 36, Qt.AlignmentFlag.AlignHCenter, score_text)

        painter.setFont(QFont("Consolas", 11))
        footer = f"steps={s.steps}  target={s.target_score}  event={s.event}"
        painter.drawText(12, field_h - 16, footer)
        painter.end()


class PongLiveRenderer:
    """Non-blocking renderer for training/evaluation loops."""

    def __init__(self, title: str = "Pong Live", fps: int = 60) -> None:
        _require_pyqt6()
        self.app = _qt_app()
        self.canvas = PongCanvas(title=title)
        self.canvas.show()
        self.fps = max(1, int(fps))
        self._min_dt = 1.0 / float(self.fps)
        self._last_t = 0.0

    def update(self, snapshot: PongSnapshot) -> None:
        now = time.monotonic()
        if now - self._last_t < self._min_dt:
            return
        self._last_t = now
        self.canvas.set_snapshot(snapshot)
        self.app.processEvents()

    def close(self) -> None:
        self.canvas.close()
        self.app.processEvents()


class HumanVsAgentWindow(QWidget):
    def __init__(
        self,
        env: PongEnv,
        agent,
        human_side: str = "left",
        fps: int = 60,
        deterministic: bool = True,
        episodes: int = 1,
    ) -> None:
        _require_pyqt6()
        if human_side not in {"left", "right"}:
            raise ValueError("human_side must be 'left' or 'right'.")

        self.app = _qt_app()
        super().__init__()
        self.env = env
        self.agent = agent
        self.human_side = human_side
        self.deterministic = deterministic
        self.target_episodes = max(1, int(episodes))
        self.completed_episodes = 0

        self.canvas = PongCanvas(title=f"Human vs Agent ({human_side})")
        self.canvas.resize(980, 560)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.canvas.installEventFilter(self)
        self.canvas.show()
        self.canvas.activateWindow()
        self.canvas.setFocus()

        self.human_up_pressed = False
        self.human_down_pressed = False
        self.last_obs_left, self.last_obs_right = self.env.reset()
        self.canvas.set_snapshot(self.env.snapshot())

        self.left_wins = 0
        self.right_wins = 0
        self.draws = 0

        self.timer = QTimer(self)
        interval_ms = max(1, int(1000 / max(1, int(fps))))
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self._tick)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        key = event.key()
        if key in {Qt.Key.Key_W, Qt.Key.Key_Up}:
            self.human_up_pressed = True
        elif key in {Qt.Key.Key_S, Qt.Key.Key_Down}:
            self.human_down_pressed = True

    def keyReleaseEvent(self, event) -> None:  # type: ignore[override]
        key = event.key()
        if key in {Qt.Key.Key_W, Qt.Key.Key_Up}:
            self.human_up_pressed = False
        elif key in {Qt.Key.Key_S, Qt.Key.Key_Down}:
            self.human_down_pressed = False

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if watched is self.canvas and event.type() == QEvent.Type.KeyPress:
            self.keyPressEvent(event)
            return True
        if watched is self.canvas and event.type() == QEvent.Type.KeyRelease:
            self.keyReleaseEvent(event)
            return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.timer.isActive():
            self.timer.stop()
        event.accept()

    def _human_action(self) -> int:
        if self.human_up_pressed and not self.human_down_pressed:
            return PongEnv.UP
        if self.human_down_pressed and not self.human_up_pressed:
            return PongEnv.DOWN
        return PongEnv.STAY

    def _tick(self) -> None:
        human_action = self._human_action()

        if self.human_side == "left":
            left_action = human_action
            right_action = self.agent.act(self.last_obs_right, deterministic=self.deterministic)
        else:
            left_action = self.agent.act(self.last_obs_left, deterministic=self.deterministic)
            right_action = human_action

        (self.last_obs_left, self.last_obs_right), _rewards, done, _info = self.env.step(left_action, right_action)
        self.canvas.set_snapshot(self.env.snapshot())

        if not done:
            return

        if self.env.score_left > self.env.score_right:
            self.left_wins += 1
        elif self.env.score_right > self.env.score_left:
            self.right_wins += 1
        else:
            self.draws += 1

        self.completed_episodes += 1
        if self.completed_episodes >= self.target_episodes:
            self.timer.stop()
            self.canvas.close()
            self.app.quit()
            return

        self.last_obs_left, self.last_obs_right = self.env.reset()
        self.canvas.set_snapshot(self.env.snapshot())

    def run(self) -> dict[str, float]:
        self.timer.start()
        self.app.exec()
        played = max(1, self.completed_episodes)
        return {
            "episodes": float(self.completed_episodes),
            "left_win_rate": float(self.left_wins / played),
            "right_win_rate": float(self.right_wins / played),
            "draw_rate": float(self.draws / played),
        }


def run_human_vs_agent(
    env: PongEnv,
    agent,
    *,
    human_side: str = "left",
    fps: int = 60,
    deterministic: bool = True,
    episodes: int = 1,
) -> dict[str, float]:
    window = HumanVsAgentWindow(
        env=env,
        agent=agent,
        human_side=human_side,
        fps=fps,
        deterministic=deterministic,
        episodes=episodes,
    )
    return window.run()
