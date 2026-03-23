import { AbsoluteFill, Audio, Sequence } from "remotion";
import { SceneFrame } from "./components/SceneFrame";
import { Subtitles } from "./components/Subtitles";
import { TransitionEffect } from "./components/Transition";

interface FrameData {
  frameId: number;
  imagePath: string;
  audioPath: string;
  durationSeconds: number;
  narrationText: string;
  transition: string;
}

export interface StoryVideoProps {
  fps: number;
  width: number;
  height: number;
  frames: FrameData[];
}

const TRANSITION_DURATION = 0.5;

export const StoryVideo: React.FC<StoryVideoProps> = ({ fps, frames }) => {
  let currentFrame = 0;

  return (
    <AbsoluteFill style={{ backgroundColor: "black" }}>
      {frames.map((frame, index) => {
        const durationFrames = Math.round(frame.durationSeconds * fps);
        const startFrame = currentFrame;
        currentFrame += durationFrames;

        return (
          <Sequence
            key={frame.frameId}
            from={startFrame}
            durationInFrames={durationFrames}
          >
            <SceneFrame
              imagePath={frame.imagePath}
              durationFrames={durationFrames}
            />
            <Subtitles
              text={frame.narrationText}
              durationFrames={durationFrames}
            />
            <Audio src={frame.audioPath} />
            {index > 0 && (
              <TransitionEffect
                type={frame.transition}
                durationFrames={Math.round(TRANSITION_DURATION * fps)}
              />
            )}
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
