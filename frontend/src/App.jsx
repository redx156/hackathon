import { useState } from 'react';
import { RoleSelection } from './pages/RoleSelection';
import { AuthenticationChoice } from './pages/AuthenticationChoice';
import { DoctorRegistration } from './pages/DoctorRegistration';
import { PatientRegistration } from './pages/PatientRegistration';
import { LoginScreen } from './pages/LoginScreen';
import { DoctorDashboard } from './pages/DoctorDashboard';
import { PatientDashboard } from './pages/PatientDashboard';
import { DoctorResults } from './pages/DoctorResults';
import { PatientResults } from './pages/PatientResults';
import { AIReviewInProgress } from './pages/AIReviewInProgress';
import DotGrid from './components/DotGrid';
import { analyzeXRay, APIError } from './services/api';
import { useAppContext } from './context/AppContext';

export default function App() {
    // Get shared state from context
    const { doctors, addDoctor, addSubmission, markAsReviewed, getSubmissionsForDoctor, getDashboardCounts } = useAppContext();

    const [currentScreen, setCurrentScreen] = useState('role-selection');
    const [selectedRole, setSelectedRole] = useState(null);
    const [currentUser, setCurrentUser] = useState(null);
    const [uploadedXRayUrl, setUploadedXRayUrl] = useState(null);
    const [pendingGoogleAuth, setPendingGoogleAuth] = useState(false);

    // API integration state
    const [selectedFile, setSelectedFile] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [analysisError, setAnalysisError] = useState(null);
    const [currentSubmissionId, setCurrentSubmissionId] = useState(null);

    const handleRoleSelection = (role) => {
        setSelectedRole(role);
        if (pendingGoogleAuth) {
            setPendingGoogleAuth(false);
            sessionStorage.setItem('googleUserRole', role);
            handleLoginSuccess(role);
            return;
        }
        setCurrentScreen('auth-choice');
    };

    const handleRegister = () => setCurrentScreen('register');
    const handleLogin = () => setCurrentScreen('login');

    // Registration success - store user data and add to context if doctor
    const handleRegistrationSuccess = (userData, isDoctor) => {
        if (userData) {
            sessionStorage.setItem('registeredUser', JSON.stringify(userData));
            // If doctor registration, add to doctors list
            if (isDoctor) {
                const newDoctor = addDoctor(userData);
                sessionStorage.setItem('registeredDoctorId', newDoctor.id);
            }
        }
        setCurrentScreen('login');
    };

    // Login success - use actual user data from form or registration
    const handleLoginSuccess = (role, userData) => {
        const registeredUser = sessionStorage.getItem('registeredUser');
        const parsedUser = registeredUser ? JSON.parse(registeredUser) : null;

        if (role === 'doctor') {
            // Find doctor in context by email (from login form or registration)
            const email = userData?.email || parsedUser?.email;
            const existingDoctor = email ? doctors.find(d => d.email === email) : null;

            if (existingDoctor) {
                // Use the existing doctor's ID from context - this matches what patients selected
                setCurrentUser({ id: existingDoctor.id, name: existingDoctor.name, email: existingDoctor.email, role: 'doctor' });
            } else {
                // Fallback for demo login or unregistered doctor
                const name = userData?.name || parsedUser?.fullName || 'Dr. Sarah Johnson';
                setCurrentUser({ id: '1', name, email: email || 'demo@hospital.com', role: 'doctor' });
            }
            setCurrentScreen('doctor-dashboard');
        } else {
            const name = userData?.name || parsedUser?.fullName || 'John Anderson';
            const doctorId = parsedUser?.selectedDoctorId;
            const assignedDoctor = doctorId
                ? doctors.find(d => d.id === doctorId) || doctors[0]
                : doctors[0];
            setCurrentUser({ id: `patient-${Date.now()}`, name, assignedDoctor, role: 'patient' });
            setCurrentScreen('patient-dashboard');
        }

        sessionStorage.removeItem('registeredUser');
    };

    const handleGoogleSignIn = () => {
        const storedRole = sessionStorage.getItem('googleUserRole');
        if (storedRole) {
            handleLoginSuccess(storedRole);
            return;
        }
        if (selectedRole) {
            sessionStorage.setItem('googleUserRole', selectedRole);
            handleLoginSuccess(selectedRole);
            return;
        }
        setPendingGoogleAuth(true);
        setCurrentScreen('role-selection');
    };

    const handleLogout = () => {
        setCurrentUser(null);
        setSelectedRole(null);
        setUploadedXRayUrl(null);
        setSelectedFile(null);
        setAnalysisResult(null);
        setAnalysisError(null);
        setCurrentSubmissionId(null);
        sessionStorage.removeItem('googleUserRole');
        setCurrentScreen('role-selection');
    };

    const handleBackToRoleSelection = () => {
        setSelectedRole(null);
        setCurrentScreen('role-selection');
    };

    const handleBackToAuthChoice = () => setCurrentScreen('auth-choice');
    const handleBackToDoctorDashboard = () => {
        setAnalysisResult(null);
        setAnalysisError(null);
        setSelectedFile(null);
        setCurrentSubmissionId(null);
        setCurrentScreen('doctor-dashboard');
    };

    // REAL API CALL - Doctor uploads X-ray for analysis
    const handleDoctorAnalyzeXRay = async (file) => {
        if (!file || isAnalyzing) return;

        setSelectedFile(file);
        setUploadedXRayUrl(URL.createObjectURL(file));
        setIsAnalyzing(true);
        setAnalysisError(null);
        setCurrentScreen('ai-review-in-progress');

        try {
            const result = await analyzeXRay(file);
            setAnalysisResult(result);
            setCurrentScreen('doctor-results');
        } catch (error) {
            console.error('Analysis failed:', error);
            if (error instanceof APIError) {
                setAnalysisError(error.message);
            } else {
                setAnalysisError('An unexpected error occurred. Please try again.');
            }
            setCurrentScreen('doctor-results');
        } finally {
            setIsAnalyzing(false);
        }
    };

    // REAL API CALL - Patient uploads X-ray (creates submission for doctor)
    const handlePatientUploadXRay = async (file, symptoms) => {
        if (!file || isAnalyzing) return;

        const xrayUrl = URL.createObjectURL(file);
        setSelectedFile(file);
        setUploadedXRayUrl(xrayUrl);
        setIsAnalyzing(true);
        setAnalysisError(null);
        setCurrentScreen('ai-review-in-progress');

        try {
            const result = await analyzeXRay(file);

            // Create submission for the assigned doctor
            const submission = addSubmission({
                patientName: currentUser.name,
                patientId: currentUser.id,
                assignedDoctorId: currentUser.assignedDoctor.id,
                xrayImageUrl: xrayUrl,
                heatmapImageUrl: result.heatmapDataUri,
                diagnosis: result.prediction,
                confidence: result.confidencePercent,
                riskLevel: result.riskLevel?.toLowerCase().includes('high') ? 'high' :
                    result.riskLevel?.toLowerCase().includes('medium') ? 'medium' : 'low',
                aiObservation: result.note,
                lowImageQuality: result.lowImageQuality,
                symptoms: symptoms,
            });

            setCurrentSubmissionId(submission.id);
            setAnalysisResult({ ...result, symptoms });
            setCurrentScreen('patient-results');
        } catch (error) {
            console.error('Analysis failed:', error);
            if (error instanceof APIError) {
                setAnalysisError(error.message);
            } else {
                setAnalysisError('An unexpected error occurred. Please try again.');
            }
            setCurrentScreen('patient-results');
        } finally {
            setIsAnalyzing(false);
        }
    };

    // Doctor reviews a case from dashboard
    const handleReviewCase = (submissionId) => {
        setCurrentSubmissionId(submissionId);
        // Get submission data for display
        const doctorSubmissions = getSubmissionsForDoctor(currentUser.id);
        const submission = doctorSubmissions.find(s => s.id === submissionId);
        if (submission) {
            setUploadedXRayUrl(submission.xrayImageUrl);
            setAnalysisResult({
                prediction: submission.diagnosis,
                confidencePercent: submission.confidence,
                heatmapDataUri: submission.heatmapImageUrl,
                riskLevel: submission.riskLevel,
                note: submission.aiObservation,
                lowImageQuality: submission.lowImageQuality,
                symptoms: submission.symptoms,
            });
        } else {
            setAnalysisResult(null);
        }
        setCurrentScreen('doctor-results');
    };

    const handleAnalysisComplete = () => setCurrentScreen('doctor-results');

    const handleMarkReviewed = (notes) => {
        if (currentSubmissionId) {
            markAsReviewed(currentSubmissionId, notes);
        }
        handleBackToDoctorDashboard();
    };

    const handleBackToDashboard = () => {
        setAnalysisResult(null);
        setAnalysisError(null);
        setSelectedFile(null);
        setCurrentSubmissionId(null);
        setCurrentScreen('patient-dashboard');
    };

    // Build case data for doctor results
    const getCaseData = () => {
        if (analysisResult) {
            const doctorSubmissions = getSubmissionsForDoctor(currentUser?.id);
            const submission = currentSubmissionId
                ? doctorSubmissions.find(s => s.id === currentSubmissionId)
                : null;

            return {
                id: currentSubmissionId || 'api-result',
                patientName: submission?.patientName || currentUser?.name || 'Unknown Patient',
                submissionDate: submission?.submissionDate || new Date().toISOString().split('T')[0],
                xrayImageUrl: uploadedXRayUrl,
                heatmapImageUrl: analysisResult.heatmapDataUri,
                diagnosis: analysisResult.prediction,
                confidence: analysisResult.confidencePercent,
                riskLevel: analysisResult.riskLevel?.toLowerCase().includes('high') ? 'high' :
                    analysisResult.riskLevel?.toLowerCase().includes('medium') ? 'medium' : 'low',
                aiObservation: analysisResult.note,
                lowImageQuality: analysisResult.lowImageQuality,
                patientSymptoms: analysisResult.symptoms || null,
                status: submission?.status || 'pending',
            };
        }
        return null;
    };

    return (
        <>
            <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', zIndex: -1, pointerEvents: 'none' }}>
                <DotGrid dotSize={5} gap={15} baseColor="#f0efef" activeColor="#8f7eec" proximity={120} shockRadius={250} shockStrength={5} returnDuration={1.5} />
            </div>

            {currentScreen === 'role-selection' && <RoleSelection onContinue={handleRoleSelection} />}
            {currentScreen === 'auth-choice' && selectedRole && <AuthenticationChoice role={selectedRole} onRegister={handleRegister} onLogin={handleLogin} onBack={handleBackToRoleSelection} />}
            {currentScreen === 'register' && selectedRole === 'doctor' && <DoctorRegistration onSuccess={(data) => handleRegistrationSuccess(data, true)} onBack={handleBackToAuthChoice} />}
            {currentScreen === 'register' && selectedRole === 'patient' && <PatientRegistration onSuccess={(data) => handleRegistrationSuccess(data, false)} onBack={handleBackToAuthChoice} />}
            {currentScreen === 'login' && selectedRole && <LoginScreen role={selectedRole} onLoginSuccess={handleLoginSuccess} onGoogleSignIn={handleGoogleSignIn} onBack={handleBackToAuthChoice} />}

            {currentScreen === 'doctor-dashboard' && currentUser && (
                <DoctorDashboard
                    doctorName={currentUser.name}
                    doctorId={currentUser.id}
                    onReviewCase={handleReviewCase}
                    onAnalyzeXRay={handleDoctorAnalyzeXRay}
                    isAnalyzing={isAnalyzing}
                    onLogout={handleLogout}
                />
            )}

            {currentScreen === 'patient-dashboard' && currentUser && currentUser.assignedDoctor && (
                <PatientDashboard
                    patientName={currentUser.name}
                    assignedDoctor={currentUser.assignedDoctor}
                    onUploadXRay={handlePatientUploadXRay}
                    isAnalyzing={isAnalyzing}
                    onLogout={handleLogout}
                />
            )}

            {currentScreen === 'doctor-results' && currentUser && getCaseData() && (
                <DoctorResults
                    caseData={getCaseData()}
                    doctorName={currentUser.name}
                    error={analysisError}
                    onMarkReviewed={handleMarkReviewed}
                    onBack={handleBackToDoctorDashboard}
                />
            )}

            {currentScreen === 'ai-review-in-progress' && (
                <AIReviewInProgress
                    onAnalysisComplete={handleAnalysisComplete}
                    isRealAnalysis={isAnalyzing}
                />
            )}

            {currentScreen === 'patient-results' && currentUser && currentUser.assignedDoctor && (
                <PatientResults
                    patientName={currentUser.name}
                    xrayImageUrl={uploadedXRayUrl}
                    analysisResult={analysisResult}
                    error={analysisError}
                    assignedDoctorName={currentUser.assignedDoctor.name}
                    onBack={handleBackToDashboard}
                />
            )}
        </>
    );
}
