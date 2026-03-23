import { AbsoluteFill, Img, useCurrentFrame, interpolate } from "remotion";

interface SceneFrameProps {
  imagePath: string;
  durationFrames: number;
}

export const SceneFrame: React.FC<SceneFrameProps> = ({
  imagePath,
  durationFrames,
}) => {
  const frame = useCurrentFrame();

  const scale = interpolate(frame, [0, durationFrames], [1.0, 1.1], {
    extrapolateRight: "clamp",
  });

  const translateX = interpolate(frame, [0, durationFrames], [0, -20], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill>
      <Img
        src={imagePath}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
          transform: `scale(${scale}) translateX(${translateX}px)`,
        }}
      />
    </AbsoluteFill>
  );
};
