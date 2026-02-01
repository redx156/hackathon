import { useEffect, useState, useRef } from 'react';

export function AIReviewInProgress({ onAnalysisComplete, isRealAnalysis }) {
    const [progress, setProgress] = useState(0);
    const [currentStep, setCurrentStep] = useState(0);
    const hasCompletedRef = useRef(false);

    const analysisSteps = [
        'Preprocessing image...',
        'Detecting lung regions...',
        'Analyzing tissue patterns...',
        'Identifying abnormalities...',
        'Generating heatmap...',
        'Compiling results...'
    ];

    useEffect(() => {
        // For real analysis, we just show the animation
        // The completion is handled by the API call in App.jsx
        if (isRealAnalysis) {
            // Show continuous loading animation
            const progressInterval = setInterval(() => {
                setProgress(prev => {
                    // Loop between 0-90% for real analysis (never "complete" on its own)
                    if (prev >= 90) return 10;
                    return prev + 2;
                });
            }, 150);

            const stepInterval = setInterval(() => {
                setCurrentStep(prev => (prev + 1) % analysisSteps.length);
            }, 1200);

            return () => {
                clearInterval(progressInterval);
                clearInterval(stepInterval);
            };
        }

        // For mock analysis (fallback), run timer-based completion
        const progressInterval = setInterval(() => {
            setProgress(prev => prev >= 100 ? (clearInterval(progressInterval), 100) : prev + 2);
        }, 100);

        const stepInterval = setInterval(() => {
            setCurrentStep(prev => prev >= analysisSteps.length - 1 ? (clearInterval(stepInterval), prev) : prev + 1);
        }, 800);

        const completeTimeout = setTimeout(() => {
            if (!hasCompletedRef.current) {
                hasCompletedRef.current = true;
                onAnalysisComplete();
            }
        }, 5000);

        return () => {
            clearInterval(progressInterval);
            clearInterval(stepInterval);
            clearTimeout(completeTimeout);
        };
    }, [onAnalysisComplete, isRealAnalysis]);

    return (
        <div className="min-h-screen bg-transparent flex items-center justify-center p-4">
            <div className="max-w-md w-full bg-white rounded-2xl shadow-xl p-8 text-center">
                <div className="mb-8">
                    <div className="w-24 h-24 bg-blue-100 rounded-full mx-auto flex items-center justify-center animate-pulse">
                        <span className="text-4xl">🤖</span>
                    </div>
                </div>

                <h1 className="text-2xl font-bold text-gray-900 mb-2">AI Analysis in Progress</h1>
                <p className="text-gray-600 mb-8">Our AI is examining your X-ray image</p>

                <div className="mb-6">
                    <div className="w-full bg-gray-200 rounded-full h-3 mb-2">
                        <div
                            className="bg-blue-600 h-3 rounded-full transition-all duration-300 ease-out"
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                    <div className="text-sm text-gray-500">
                        {isRealAnalysis ? 'Processing...' : `${progress}% Complete`}
                    </div>
                </div>

                <div className="bg-gray-50 rounded-lg p-4 mb-6">
                    <div className="flex items-center justify-center space-x-2">
                        <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" />
                        <span className="text-gray-700">{analysisSteps[currentStep]}</span>
                    </div>
                </div>

                <div className="text-left space-y-2">
                    {analysisSteps.map((step, index) => (
                        <div
                            key={index}
                            className={`flex items-center text-sm ${index < currentStep ? 'text-green-600' :
                                    index === currentStep ? 'text-blue-600 font-medium' :
                                        'text-gray-400'
                                }`}
                        >
                            <span className="mr-2">
                                {index < currentStep ? '✓' : index === currentStep ? '●' : '○'}
                            </span>
                            {step}
                        </div>
                    ))}
                </div>

                <p className="text-xs text-gray-400 mt-8">
                    {isRealAnalysis ? 'Waiting for server response...' : 'This usually takes less than a minute'}
                </p>
            </div>
        </div>
    );
}
