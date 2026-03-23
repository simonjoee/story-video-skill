import { AbsoluteFill, useCurrentFrame, interpolate } from "remotion";

interface SubtitlesProps {
  text: string;
  durationFrames: number;
}

export const Subtitles: React.FC<SubtitlesProps> = ({
  text,
  durationFrames,
}) => {
  const frame = useCurrentFrame();

  const opacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        justifyContent: "flex-end",
        alignItems: "center",
        paddingBottom: 60,
      }}
    >
      <div
        style={{
          backgroundColor: "rgba(0, 0, 0, 0.6)",
          padding: "12px 24px",
          borderRadius: 8,
          opacity,
          maxWidth: "80%",
        }}
      >
        <p
          style={{
            color: "white",
            fontSize: 36,
            fontFamily: "sans-serif",
            textAlign: "center",
            margin: 0,
            textShadow: "1px 1px 2px rgba(0,0,0,0.8)",
            lineHeight: 1.4,
          }}
        >
          {text}
        </p>
      </div>
    </AbsoluteFill>
  );
};
