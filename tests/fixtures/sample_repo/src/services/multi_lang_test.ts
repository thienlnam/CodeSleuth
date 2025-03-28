// TypeScript/JavaScript style functions and classes
export class UserManager {
    private users: Map<string, User>;

    constructor() {
        this.users = new Map();
    }

    // Standard method
    public async createUser(username: string): Promise<User> {
        return new User(username);
    }
}

// Arrow function
const validateUser = (user: User): boolean => {
    return user.isValid();
};

// Function declaration
function processUser(user: User) {
    return user.process();
}

// Async function
async function fetchUserData(id: string) {
    return await api.getUser(id);
}

// Interface (should not be detected as class)
interface User {
    id: string;
    name: string;
    isValid(): boolean;
    process(): void;
}

// Abstract class
export abstract class BaseService {
    protected abstract init(): void;
}

// Generic class
export class DataStore<T> {
    private items: T[];

    constructor() {
        this.items = [];
    }
}
