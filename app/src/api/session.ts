import type { User } from '../types'

const KEY = 'calyx_user'
const TOKEN_KEY = 'calyx_token'

export function getSession(): User | null {
  try {
    const raw = sessionStorage.getItem(KEY)
    return raw ? (JSON.parse(raw) as User) : null
  } catch {
    return null
  }
}

// The token is stored separately so refreshing the profile (which comes
// back without a token) never logs the user out.
export function setSession(user: User & { token?: string }): void {
  const { token, ...profile } = user
  sessionStorage.setItem(KEY, JSON.stringify(profile))
  if (token) sessionStorage.setItem(TOKEN_KEY, token)
}

export function getToken(): string | null {
  return sessionStorage.getItem(TOKEN_KEY)
}

export function clearSession(): void {
  sessionStorage.removeItem(KEY)
  sessionStorage.removeItem(TOKEN_KEY)
}
