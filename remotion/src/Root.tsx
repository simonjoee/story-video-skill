import { Composition } from "remotion";
import { StoryVideo, StoryVideoProps } from "./StoryVideo";

export const Root: React.FC = () => {
  return (
    <Composition
      id="StoryVideo"
      component={StoryVideo as unknown as React.ComponentType<Record<string, unknown>>}
      durationInFrames={300}
      fps={30}
      width={1920}
      height={1080}
      defaultProps={{
        fps: 30,
        width: 1920,
        height: 1080,
        frames: [],
      } as StoryVideoProps}
      calculateMetadata={({ props }) => {
        const p = props as unknown as StoryVideoProps;
        const totalFrames = p.frames.reduce(
          (sum: number, f: { durationSeconds: number }) => sum + Math.round(f.durationSeconds * p.fps),
          0
        );
        return {
          durationInFrames: Math.max(totalFrames, 1),
          fps: p.fps,
          width: p.width,
          height: p.height,
        };
      }}
    />
  );
};
