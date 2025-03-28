import React, { useState, useEffect } from 'react';
import { User } from './types';

interface AppProps {
    initialUser?: User;
}

export const App: React.FC<AppProps> = ({ initialUser }) => {
    const [user, setUser] = useState<User | undefined>(initialUser);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!user) {
            fetchUser();
        }
    }, [user]);

    const fetchUser = async () => {
        try {
            setLoading(true);
            const response = await fetch('/api/users/1');
            if (!response.ok) {
                throw new Error('Failed to fetch user');
            }
            const userData = await response.json();
            setUser(userData);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setLoading(false);
        }
    };

    const handleLogin = async (username: string, password: string) => {
        try {
            setLoading(true);
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username, password }),
            });

            if (!response.ok) {
                throw new Error('Login failed');
            }

            const userData = await response.json();
            setUser(userData);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Login failed');
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return <div>Loading...</div>;
    }

    if (error) {
        return <div>Error: {error}</div>;
    }

    return (
        <div className="app">
            <header>
                <h1>Welcome {user?.username || 'Guest'}</h1>
            </header>
            <main>
                {!user ? (
                    <LoginForm onSubmit={handleLogin} />
                ) : (
                    <UserProfile user={user} />
                )}
            </main>
        </div>
    );
};

interface LoginFormProps {
    onSubmit: (username: string, password: string) => Promise<void>;
}

const LoginForm: React.FC<LoginFormProps> = ({ onSubmit }) => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        onSubmit(username, password);
    };

    return (
        <form onSubmit={handleSubmit}>
            <div>
                <label htmlFor="username">Username:</label>
                <input
                    type="text"
                    id="username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    required
                />
            </div>
            <div>
                <label htmlFor="password">Password:</label>
                <input
                    type="password"
                    id="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                />
            </div>
            <button type="submit">Login</button>
        </form>
    );
};

interface UserProfileProps {
    user: User;
}

const UserProfile: React.FC<UserProfileProps> = ({ user }) => {
    return (
        <div className="user-profile">
            <h2>Profile</h2>
            <p>Username: {user.username}</p>
            <p>Email: {user.email}</p>
            <p>
                Member since: {new Date(user.created_at).toLocaleDateString()}
            </p>
            {user.last_login && (
                <p>
                    Last login: {new Date(user.last_login).toLocaleDateString()}
                </p>
            )}
        </div>
    );
};
