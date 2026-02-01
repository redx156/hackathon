/**
 * AppContext - Shared state for doctors and patient submissions
 * 
 * Provides:
 * - List of registered doctors (starts with mock data, grows with registrations)
 * - List of patient submissions (X-ray uploads)
 * - Functions to add doctors, submissions, and mark as reviewed
 */

import { createContext, useContext, useState } from 'react';

// Initial mock doctors
const initialDoctors = [
    { id: '1', name: 'Dr. Sarah Johnson', hospital: 'City General Hospital', specialization: 'Radiology', email: 'sarah.johnson@hospital.com' },
    { id: '2', name: 'Dr. Michael Chen', hospital: 'University Medical Center', specialization: 'Pulmonology', email: 'michael.chen@hospital.com' },
    { id: '3', name: 'Dr. Emily Rodriguez', hospital: "St. Mary's Hospital", specialization: 'Internal Medicine', email: 'emily.rodriguez@hospital.com' },
    { id: '4', name: 'Dr. James Williams', hospital: 'Regional Health Center', specialization: 'Radiology', email: 'james.williams@hospital.com' },
];

// Create context
const AppContext = createContext(null);

// Provider component
export function AppProvider({ children }) {
    // Shared doctors list - grows when new doctors register
    const [doctors, setDoctors] = useState(initialDoctors);

    // Patient submissions - visible to assigned doctor
    const [submissions, setSubmissions] = useState([]);

    // Add a new doctor (called after doctor registration)
    const addDoctor = (doctorData) => {
        const newDoctor = {
            id: `doc-${Date.now()}`,
            name: doctorData.fullName,
            hospital: doctorData.hospital,
            specialization: doctorData.specialization,
            email: doctorData.email,
        };
        setDoctors(prev => [...prev, newDoctor]);
        return newDoctor;
    };

    // Add a patient submission (called after X-ray upload)
    const addSubmission = (submission) => {
        const newSubmission = {
            id: `sub-${Date.now()}`,
            ...submission,
            status: 'pending',
            submissionDate: new Date().toISOString().split('T')[0],
        };
        setSubmissions(prev => [...prev, newSubmission]);
        return newSubmission;
    };

    // Mark a submission as reviewed
    const markAsReviewed = (submissionId, doctorNotes) => {
        setSubmissions(prev => prev.map(sub =>
            sub.id === submissionId
                ? { ...sub, status: 'reviewed', doctorNotes, reviewedAt: new Date().toISOString() }
                : sub
        ));
    };

    // Get submissions for a specific doctor
    const getSubmissionsForDoctor = (doctorId) => {
        return submissions.filter(sub => sub.assignedDoctorId === doctorId);
    };

    // Get counts for doctor dashboard
    const getDashboardCounts = (doctorId) => {
        const doctorSubmissions = getSubmissionsForDoctor(doctorId);
        return {
            pending: doctorSubmissions.filter(s => s.status === 'pending').length,
            reviewed: doctorSubmissions.filter(s => s.status === 'reviewed').length,
            highRisk: doctorSubmissions.filter(s => s.riskLevel === 'high').length,
            total: doctorSubmissions.length,
        };
    };

    const value = {
        doctors,
        submissions,
        addDoctor,
        addSubmission,
        markAsReviewed,
        getSubmissionsForDoctor,
        getDashboardCounts,
    };

    return (
        <AppContext.Provider value={value}>
            {children}
        </AppContext.Provider>
    );
}

// Hook to use the context
export function useAppContext() {
    const context = useContext(AppContext);
    if (!context) {
        throw new Error('useAppContext must be used within AppProvider');
    }
    return context;
}
