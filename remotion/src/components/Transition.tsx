import { AbsoluteFill, useCurrentFrame, interpolate } from "remotion";

interface TransitionProps {
  type: string;
  durationFrames: number;
}

export const TransitionEffect: React.FC<TransitionProps> = ({
  type,
  durationFrames,
}) => {
  const frame = useCurrentFrame();

  if (type === "cut") return null;

  const opacity = interpolate(frame, [0, durationFrames], [1, 0], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: type === "fade" ? "black" : "transparent",
        opacity: type === "fade" ? opacity : 0,
      }}
    />
  );
};
