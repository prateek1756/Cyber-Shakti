import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { MessageSquareWarning, Loader2, ShieldCheck, ShieldAlert } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/use-toast";

interface FraudAnalysisResult {
  classification: "fraud" | "safe" | "suspicious";
  risk_score: number;
  explanations: string[];
  details: {
    nlp: { fraud_score: number; label: string; confidence: number };
    phishing_heuristics: { score: number; indicators: string[] };
    top_category: string;
    keywords: { score: number; matches: string[] };
    urls: { score: number; suspicious_urls: string[] };
  };
}

function RiskPill({ level, score }: { level: "safe" | "fraud" | "suspicious"; score: number }) {
  const styles = {
    safe: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30",
    suspicious: "bg-yellow-500/15 text-yellow-300 ring-yellow-500/30",
    fraud: "bg-red-500/15 text-red-300 ring-red-500/30",
  } as const;

  const labels = {
    safe: "Likely Safe",
    suspicious: "Suspicious",
    fraud: "Potential Fraud"
  } as const;

  return (
    <span className={cn("inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs ring-1", styles[level])}>
      {level === "safe" ? <ShieldCheck className="w-3 h-3" /> : <ShieldAlert className="w-3 h-3" />}
      <span className="font-semibold tracking-wide uppercase">{labels[level]}</span>
      <span className="text-foreground/60">{score}/100</span>
    </span>
  );
}

export default function FraudDetection() {
  const [msg, setMsg] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<FraudAnalysisResult | null>(null);
  const { toast } = useToast();

  const handleAnalyze = async () => {
    if (!msg.trim()) return;

    setIsLoading(true);
    setResult(null);

    try {
      const response = await fetch("/api/fraud/detect", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: msg }),
      });

      if (!response.ok) {
        throw new Error("Failed to analyze message");
      }

      const responseData = await response.json();

      if (!responseData.success) {
        throw new Error(responseData.error || "Analysis failed");
      }

      setResult(responseData.data);
    } catch (error) {
      console.error("Analysis error:", error);
      const detail = error instanceof Error ? error.message : "Network failure";
      toast({
        title: "Error",
        description: `Analysis failed: ${detail}. Please check Vercel logs for Python errors.`,
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative">
      <div className="container py-8">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-extrabold tracking-tight md:text-5xl">
            Fraud Message Detection
          </h1>
          <p className="mx-auto mt-2 max-w-2xl text-muted-foreground">
            AI-powered analysis to detect social engineering and payment scams
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <Card className="bg-gradient-to-b from-card/60 to-card/30">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MessageSquareWarning className="h-5 w-5 text-primary" />
                Analyze Message Content
              </CardTitle>
              <CardDescription>
                Paste suspicious messages, emails, or texts to check for fraud indicators using our advanced AI engine.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <textarea
                value={msg}
                onChange={(e) => setMsg(e.target.value)}
                rows={8}
                placeholder="Paste your message here... Example: 'URGENT! Your account will be suspended. Click this link immediately and provide your OTP to verify...'"
                className="w-full rounded-md border bg-background/60 p-3 text-sm outline-none ring-offset-background placeholder:text-muted-foreground/60 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 resize-y"
              />

              <div className="flex justify-end">
                <Button
                  onClick={handleAnalyze}
                  disabled={isLoading || !msg.trim()}
                  className="w-full md:w-auto"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    "Analyze Message"
                  )}
                </Button>
              </div>

              {result && (
                <div className="rounded-md border p-6 space-y-4 bg-background/40 animate-in fade-in slide-in-from-bottom-2">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div>
                      <h3 className="text-lg font-semibold mb-1">Analysis Result</h3>
                      <p className="text-sm text-muted-foreground">
                        Based on AI, keywords, and link analysis
                      </p>
                    </div>
                    <RiskPill level={result.classification} score={result.risk_score} />
                  </div>

                  {/* Progress Bar Visual */}
                  <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                    <div
                      className={cn("h-full transition-all duration-1000 ease-out",
                        result.risk_score < 40 ? "bg-emerald-500" :
                          result.risk_score < 70 ? "bg-yellow-500" : "bg-red-500"
                      )}
                      style={{ width: `${result.risk_score}%` }}
                    />
                  </div>

                  <div className="grid md:grid-cols-2 gap-6 mt-4">
                    <div>
                      <p className="text-sm font-medium mb-2">Key Findings:</p>
                      {result.explanations.length > 0 ? (
                        <ul className="list-inside list-disc text-sm text-muted-foreground space-y-1">
                          {result.explanations.map((reason, i) => (
                            <li key={i}>{reason}</li>
                          ))}
                        </ul>
                      ) : (
                        <p className="text-sm text-muted-foreground">No specific threats detected.</p>
                      )}
                    </div>

                    <div className="text-sm space-y-2 border-l pl-4 border-border/50">
                      <p className="font-medium text-muted-foreground">Detailed Breakdown:</p>
                      <div className="flex justify-between">
                        <span>AI Suspicion:</span>
                        <span className={cn(result.details.nlp.fraud_score > 50 ? "text-red-400" : "text-emerald-400")}>
                          {result.details.nlp.fraud_score.toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Phishing Heuristics:</span>
                        <span className={cn(result.details.phishing_heuristics.score > 40 ? "text-red-400" : "text-emerald-400")}>
                          {result.details.phishing_heuristics.score}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Keyword Risk:</span>
                        <span className={cn(result.details.keywords.score > 50 ? "text-red-400" : "text-emerald-400")}>
                          {result.details.keywords.score}%
                        </span>
                      </div>
                      <div className="flex justify-between font-medium pt-2 border-t border-border/30">
                        <span>Scam Category:</span>
                        <span className="text-primary capitalize">
                          {result.details.top_category.replace('_', ' ')}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
