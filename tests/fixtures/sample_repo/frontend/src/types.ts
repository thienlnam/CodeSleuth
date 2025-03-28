export interface User {
    id: number;
    username: string;
    email: string;
    is_active: boolean;
    created_at: string;
    last_login: string | null;
}
